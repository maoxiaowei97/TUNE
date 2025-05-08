import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.distributions as dist
from torch.nn import functional as F
class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):

        return self.mlp(x)


class MAPEWithZeroMaskingLoss(nn.Module):
    def __init__(self):
        super(MAPEWithZeroMaskingLoss, self).__init__()

    def forward(self, predict, label):
        mask = label != 0
        masked_predict = predict[mask]
        masked_label = label[mask]

        loss = torch.abs((masked_predict - masked_label) / masked_label)
        return loss.mean()

class MGUQ_network(torch.nn.Module):

    def __init__(self, args):
        super(MGUQ_network, self).__init__()

        id_embed_dim = args.id_embed_dim
        slice_dims = args.slice_dim
        slice_embed_dim =  args.id_embed_dim
        mlp_out_dim = args.E_U
        n_embed = args.E_U
        segment_dims =  args.segment_dim

        self.dropout = args.dropout
        self.hidden_dim = args.E_U
        self.distribution_embed = nn.Linear(36, n_embed)
        self.c_f = args.c_f

        self.segment_embedding = nn.Embedding(segment_dims, id_embed_dim)

        self.slice_embedding = nn.Embedding(slice_dims, slice_embed_dim)
        self.all_mlp = nn.Sequential(
            nn.Linear(id_embed_dim + slice_embed_dim + n_embed , mlp_out_dim),
            nn.ReLU(),
            nn.Linear(mlp_out_dim, mlp_out_dim),
        )
        self.dec_segment_mean = nn.Linear(self.hidden_dim, 1)
        self.gru = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, dropout=self.dropout) # GRU
        self.dec_segment_diag = nn.Linear(self.hidden_dim, 1)
        self.dec_low_rank = nn.Linear(self.hidden_dim, 16)
        self.mape_loss = MAPEWithZeroMaskingLoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, xs,  route_segment_travel_time_distribution, number_of_roadsegments, start_ts,  segment_travel_time_label, total_duration, device):

        all_id_embedding = self.segment_embedding(xs.to(device))
        all_slice_embedding = self.slice_embedding(
            (start_ts).unsqueeze(1).repeat(1, xs.shape[1]).long().to(device))  # start_ts 2min -> 10 min
        all_real = self.distribution_embed(route_segment_travel_time_distribution.to(device))  # [bs, seq, n_embd]
        all_input = torch.cat([all_id_embedding, all_slice_embedding, all_real], dim=2)
        recurrent_input = self.all_mlp(all_input)
        packed_all_input = pack_padded_sequence(recurrent_input, number_of_roadsegments.reshape(-1).cpu(), enforce_sorted=False, batch_first=True)
        out_segment, _ = self.gru(packed_all_input)
        out_segment, _ = pad_packed_sequence(out_segment, batch_first=True)

        # decode mean and var of each segment
        segments_mean = self.dec_segment_mean(out_segment)
        raw_diag = self.dec_segment_diag(out_segment).squeeze(-1)
        raw_factor = self.dec_low_rank(out_segment)
        B, N_valid, _ = segments_mean.shape[0], segments_mean.shape[1], segments_mean.shape[2]
        batch_covs = []
        log_probs = []
        lower_list = []
        upper_list = []
        confidence = self.c_f
        alpha = 1.0 - confidence
        z = dist.Normal(0, 1).icdf(torch.tensor([1.0 - alpha / 2]))[0].item()
        for b in range(B):
            n = number_of_roadsegments[b].item()  # actual seq length for this sample
            # Extract the first n positions
            diag_b = raw_diag[b, :n]  # shape (n,)
            factor_b = raw_factor[b, :n, :]  # shape (n, K)
            mean_b = segments_mean.reshape(B, -1)[b, :n]

            D = F.softplus(diag_b)  # ensure positivity
            V = factor_b  # no restriction, can be negative/positive
            # Build Cov_b = diag(D) + V V^T + eps*I
            Cov_b = torch.diag(D) + V @ V.transpose(0, 1)  # shape (n,n)
            Cov_b = Cov_b + torch.eye(n, device=Cov_b.device) * (1e-5)
            mvn_b = dist.MultivariateNormal(loc=mean_b, covariance_matrix=Cov_b)
            label_b = segment_travel_time_label[b, :n]
            log_p_b = mvn_b.log_prob(label_b.to(device))  # shape ()
            log_probs.append(log_p_b)
            batch_covs.append(Cov_b)
            """
            calc lower and upper for the sequence based on the MVN
            """
            ones = torch.ones(n, device=device)
            var_sum = ones @ Cov_b @ ones
            std_sum = torch.sqrt(var_sum)
            sum_mean = mean_b.sum()
            lower = sum_mean - z * std_sum
            upper = sum_mean + z * std_sum
            lower_list.append(lower)
            upper_list.append(upper)

        mask_indices = torch.arange(N_valid).unsqueeze(0).expand(B, -1)  # [batch_size, max_segments]
        mask = (mask_indices < number_of_roadsegments.unsqueeze(1)).unsqueeze(-1).float()
        valid_segment_mean = segments_mean[mask.bool()] # total segments in all batches
        valid_segment_label = segment_travel_time_label[:, :N_valid].unsqueeze(-1)[mask.bool()]
        loss_mean_segments = self.mae_loss(valid_segment_mean, valid_segment_label.to(device))
        predict_mean = torch.sum((segments_mean * mask.to(segments_mean.device)).squeeze(-1), dim=1) # (B,)
        loss_path_eta = self.mape_loss(predict_mean, total_duration.to(device))
        log_probs = torch.stack(log_probs, dim=0)
        nll = -log_probs.mean()

        return loss_path_eta  + loss_mean_segments + nll , predict_mean, segments_mean, batch_covs, torch.stack(lower_list, dim=0).reshape(-1), torch.stack(upper_list, dim=0).reshape(-1)


    def test(self, xs, route_segment_travel_time_distribution, number_of_roadsegments, start_ts,
                segment_travel_time_label, total_duration, device):
        all_id_embedding = self.segment_embedding(xs.to(device))
        all_slice_embedding = self.slice_embedding((start_ts).unsqueeze(1).repeat(1, xs.shape[1]).long().to(device))
        all_real = self.distribution_embed(route_segment_travel_time_distribution.to(device))  # [bs, seq, n_embd]
        all_input = torch.cat([all_id_embedding, all_slice_embedding, all_real], dim=2)

        recurrent_input = self.all_mlp(all_input)

        packed_all_input = pack_padded_sequence(recurrent_input, number_of_roadsegments.reshape(-1).cpu(),
                                                enforce_sorted=False, batch_first=True)
        out_segment, _ = self.gru(packed_all_input)
        out_segment, _ = pad_packed_sequence(out_segment, batch_first=True)

        # decode mean and var of each segment
        segments_mean = self.dec_segment_mean(out_segment)
        raw_diag = self.dec_segment_diag(out_segment).squeeze(-1)
        raw_factor = self.dec_low_rank(out_segment)
        B, N_valid, _ = segments_mean.shape[0], segments_mean.shape[1], segments_mean.shape[2]
        batch_covs = []
        log_probs = []
        lower_list = []
        upper_list = []
        confidence = self.c_f
        alpha = 1.0 - confidence
        z = dist.Normal(0, 1).icdf(torch.tensor([1.0 - alpha / 2]))[0].item()
        for b in range(B):
            n = number_of_roadsegments[b].item()  # actual seq length for this sample
            # Extract the first n positions
            diag_b = raw_diag[b, :n]  # shape (n,)
            factor_b = raw_factor[b, :n, :]  # shape (n, K)
            mean_b = segments_mean.reshape(B, -1)[b, :n]

            D = F.softplus(diag_b)  # ensure positivity
            V = factor_b  # no restriction, can be negative/positive
            # Build Cov_b = diag(D) + V V^T + eps*I
            Cov_b = torch.diag(D) + V @ V.transpose(0, 1)  # shape (n,n)
            Cov_b = Cov_b + torch.eye(n, device=Cov_b.device) * (1e-5)
            mvn_b = dist.MultivariateNormal(loc=mean_b, covariance_matrix=Cov_b)
            label_b = segment_travel_time_label[b, :n]
            log_p_b = mvn_b.log_prob(label_b.to(device))  # shape ()
            log_probs.append(log_p_b)
            batch_covs.append(Cov_b)
            """
            calc lower and upper for the sequence based on the MVN
            """
            ones = torch.ones(n, device=device)
            var_sum = ones @ Cov_b @ ones
            std_sum = torch.sqrt(var_sum)
            sum_mean = mean_b.sum()
            lower = sum_mean - z * std_sum
            upper = sum_mean + z * std_sum
            lower_list.append(lower)
            upper_list.append(upper)


        mask_indices = torch.arange(N_valid).unsqueeze(0).expand(B, -1)  # [batch_size, max_segments]

        mask = (mask_indices < number_of_roadsegments.unsqueeze(1)).unsqueeze(-1).float()

        valid_segment_mean = segments_mean[mask.bool()]  # total segments in all batches
        valid_segment_label = segment_travel_time_label[:, :N_valid].unsqueeze(-1)[mask.bool()]
        loss_mean_segments = self.mae_loss(valid_segment_mean, valid_segment_label.to(device))
        predict_mean = torch.sum((segments_mean * mask.to(segments_mean.device)).squeeze(-1), dim=1)
        loss_path_eta = self.mape_loss(predict_mean, total_duration.to(device))
        log_probs = torch.stack(log_probs, dim=0)
        nll = -log_probs.mean()

        return loss_path_eta  + loss_mean_segments + nll , predict_mean, segments_mean, batch_covs, torch.stack(
            lower_list, dim=0).reshape(-1), torch.stack(upper_list, dim=0).reshape(-1)

