import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
seperate mean and distribution
"""


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=True):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash

def numerator(qs, ks, vs):
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)

def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)

def linearized_softmax(x, query, key):
    # x: [B, N, H, D] query: [B, N, H, m], key: [B, N, H, m]
    query = query.permute(1, 0, 2, 3) # [N, B, H, m]
    key = key.permute(1, 0, 2, 3) # [N, B, H, m]
    x = x.permute(1, 0, 2, 3) # [N, B, H, D]

    z_num = numerator(query, key, x) # [N, B, H, D]
    z_den = denominator(query, key) # [N, H]

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den # # [B, N, H, D]

    return z_output

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

class linearized_attention(nn.Module):
    def __init__(self, c_in, c_out, dropout, random_feature_dim=30, tau=1.0, num_heads=4):
        super(linearized_attention, self).__init__()
        self.Wk = nn.Linear(c_in, c_out * num_heads)
        self.Wq = nn.Linear(c_in, c_out * num_heads)
        self.Wv = nn.Linear(c_in, c_out * num_heads)
        self.Wo = nn.Linear(c_out * num_heads, c_out)
        self.c_in = c_in
        self.c_out = c_out
        self.num_heads = num_heads
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU
        self.dropout = dropout

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()

    def forward(self, x):
        B, T = x.size(0), x.size(1)  # (B, T, D)
        query = self.Wq(x).reshape(-1, T, self.num_heads, self.c_out)  # (B, T, H, D)
        key = self.Wk(x).reshape(-1, T, self.num_heads, self.c_out)  # (B, T, H, D)
        x = self.Wv(x).reshape(-1, T, self.num_heads, self.c_out)  # (B, T, H, D)

        dim = query.shape[-1]  # (B, T, H, D)
        seed = torch.ceil(torch.abs(torch.sum(query) * 1e8)).to(torch.int32)
        projection_matrix = create_projection_matrix(self.random_feature_dim, dim, seed=seed).to(query.device)  # (d, m)
        query = query / math.sqrt(self.tau)
        key = key / math.sqrt(self.tau)
        query = softmax_kernel_transformation(query, True, projection_matrix)  # [B, T, H, m]
        key = softmax_kernel_transformation(key, False, projection_matrix)  # [B, T, H, m]

        x = linearized_softmax(x, query, key)

        x = self.Wo(x.flatten(-2, -1))  # (B, T, D)

        return x

class MAPEWithZeroMaskingLoss(nn.Module):
    def __init__(self):
        super(MAPEWithZeroMaskingLoss, self).__init__()

    def forward(self, predict, label):
        mask = label != 0
        masked_predict = predict[mask]
        masked_label = label[mask]

        # 计算 MAPE
        loss = torch.abs((masked_predict - masked_label) / masked_label)
        return loss.mean()

def mean_to_distribution(input_mean):
    """
    Converts mean travel times to probability distributions over 35 bins.

    Args:
        input_mean: torch.Tensor of shape [B, 1] representing mean travel times.

    Returns:
        probs: torch.Tensor of shape [B, 35], representing distributions.
    """
    B = input_mean.size(0)
    times = input_mean.clamp(max=300).view(-1)

    # Define bin edges as provided
    bin_edges = torch.cat([
        torch.arange(0, 100, 4),
        torch.tensor([100]),
        torch.arange(120, 301, 20)
    ], dim=0).to(times.device).float()

    # Digitize times into bins
    indices = torch.bucketize(times, bin_edges, right=False) - 1
    indices = indices.clamp(max=34)

    # Initialize counts
    counts = torch.zeros(B, 35, device=times.device)

    # Increment counts according to bin indices
    counts[torch.arange(B), indices] = 1

    # Since each sample has exactly one count, it's already normalized
    probs = counts.float()

    return probs


class LGDE(nn.Module):
    def __init__(self, args):
        super(LGDE, self).__init__()
        self.tau = args.tau
        self.num_layers = args.num_layers
        self.random_feature_dim = args.random_feature_dim

        self.use_residual = args.use_residual
        self.use_bn = args.use_bn
        self.use_spatial = args.use_spatial
        self.use_long = args.use_long

        self.dropout = args.dropout
        self.activation = nn.ReLU()

        self.time_num = 30 * 24 + 1
        self.week_num = args.week_num + 1

        self.ts_embedding = nn.Embedding(self.time_num, args.time_dim)
        self.dow_embedding = nn.Embedding(self.week_num, args.time_dim)

        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(args.num_nodes, args.node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        self.input_emb_layer = nn.Conv2d(55, args.hid_dim, kernel_size=(1, 1), bias=True)
        self.hidden_size = args.hid_dim

        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, args.time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, args.time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        self.W_1 = nn.Conv2d( args.time_dim * 2 + args.node_dim , args.hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(args.time_dim * 2 + args.node_dim , args.hid_dim, kernel_size=(1, 1), bias=True)

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()


        for i in range(self.num_layers):
            self.linear_conv.append(
                linearized_conv((self.hidden_size + args.time_dim) * 2 + args.node_dim , (self.hidden_size + args.time_dim) * 2 + args.node_dim  ,
                                self.dropout, self.tau, self.random_feature_dim))
            self.bn.append(nn.LayerNorm((self.hidden_size + args.time_dim) * 2 + args.node_dim))

        # self.output_layer = nn.Conv2d(args.hid_dim * 2, 35, kernel_size=(1, 1), bias=True)
        self.output_layer = nn.Linear(args.hid_dim * 2, 35, bias=True)
        self.mean_emb = nn.Linear(5, args.hid_dim)
        self.dist_emb = nn.Linear(35 * 5, args.hid_dim)

        self.kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        self.mae_loss_fn = torch.nn.L1Loss()
        self.mape_loss_fn = MAPEWithZeroMaskingLoss()
        bin_edges = np.concatenate((
            np.append(np.arange(0, 100, 4), 100),
            np.arange(120, 301, 20)
        ))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # shape: [35]
        self.input_fc =  nn.Linear((self.hidden_size + args.time_dim) * 2 + args.node_dim, args.hid_dim , bias=True)
        self.bin_centers = torch.tensor(bin_centers, dtype=torch.float32)  # [35]
        self.last_interval_linear = nn.Linear(35, 35)
        self.layer_num = 1
        self.transformer_layer = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(self.layer_num):
            self.transformer_layer.append(
                linearized_attention(args.hid_dim, args.hid_dim, self.dropout, self.random_feature_dim, self.tau))
            self.bn.append(nn.LayerNorm(args.hid_dim))

    def forward(self, input, input_d_ts, label_dist, device):
        # input: (B, N, D)
        input = input[:, :, :-1, :].clone()
        B, N, D_t, H_t = input.size()
        label_dist = label_dist.to(device)
        time_emb = self.ts_embedding(input_d_ts[:, :, 1].long().to(device))
        week_emb = self.dow_embedding(input_d_ts[:, :, 2].long().to(device))
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)
        dist_input = input[:, :, :, :-1].to(device) # [B, N, 5, 35]
        dist_input = dist_input.reshape(B, N, -1)
        dist_emb = self.dist_emb(dist_input)
        mean_input = input[:, :, :, -1].to(device)
        mean_emb = self.mean_emb(mean_input)

        x = torch.cat([dist_emb.permute(0, 2, 1).contiguous().unsqueeze(-1), mean_emb.permute(0, 2, 1).contiguous().unsqueeze(-1), node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        x = self.input_fc(x)

        x_pool = [x]

        for num in range(self.layer_num):
            residual = x  # (B*N, T/12, nhid)
            x = self.transformer_layer[num](x)  # (B*N, T/12, nhid)
            x = self.bn[num](x)
            x = x + residual  # (B*N, T/12, nhid)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=-1)  #  (B, N, dim), (B, dim*4, N, 1)

        x = self.activation(x)  # (B, dim*4, N, 1)

        x = self.output_layer(x)  # (B, N, out_dim)
        # x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        predicted_probs = F.softmax(x, dim=-1) # 和均值对应分布加权平均
        dist_last_interval = mean_to_distribution(input[:, :, 2, -1].reshape(-1, 1)).to(device)
        dist_last_interval = self.last_interval_linear(dist_last_interval)
        predicted_probs = 0.5 * predicted_probs.reshape(B*N, -1) + (0.5) * dist_last_interval

        epsilon = 1e-6  # Small constant to smooth the true distribution
        true_distribution_smooth = label_dist[:, :, :-1] + epsilon
        true_distribution_smooth_sum = true_distribution_smooth.sum(dim=-1, keepdim=True)
        true_distribution_smooth_sum = torch.clamp(true_distribution_smooth_sum, min=epsilon)
        true_distribution_smooth = true_distribution_smooth / true_distribution_smooth_sum
        predicted_probs = torch.clamp(predicted_probs, min=epsilon)
        log_predicted_probs = torch.log(predicted_probs)
        log_predicted_probs_flat = log_predicted_probs.reshape(-1, predicted_probs.size(-1))
        true_distribution_flat = true_distribution_smooth.reshape(-1, predicted_probs.size(-1))
        loss_kl = self.kl_loss_fn(log_predicted_probs_flat, true_distribution_flat)

        """
        KLDiv, and mean
        """
        label_mean = label_dist[:, :, -1]
        predicted_mean = torch.matmul(predicted_probs.reshape(B * N, -1), self.bin_centers.unsqueeze(1).to(device))
        loss_mae = self.mae_loss_fn(predicted_mean.reshape(B * N), label_mean.reshape(B * N))
        loss_mape = self.mape_loss_fn(predicted_mean.reshape(B * N), label_mean.reshape(B * N))
        loss_rmse = torch.sqrt(F.mse_loss(predicted_mean.reshape(B * N), label_mean.reshape(B * N)))

        return loss_mae, loss_kl, loss_mape, loss_rmse, predicted_probs, predicted_mean


def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def create_random_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)


def random_feature_map(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                            dim=attention_dims_t, keepdim=True)[
                    0]) + numerical_stabilizer
        )
    return data_dash


def linear_kernel(x, node_vec1, node_vec2):
    # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)  # [N, B, 1, r]
    x = x.permute(1, 0, 2, 3)  # [N, B, 1, nhid]

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)  # [N, B, 1, nhid]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)  # [N, 1]

    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
    out2 = out2.permute(1, 0, 2)
    out2 = torch.unsqueeze(out2, len(out2.shape))
    out = out1 / out2  # [B, N, 1, nhid]

    return out



class conv_approximation(nn.Module):
    def __init__(self, dropout, tau, random_feature_dim):
        super(conv_approximation, self).__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, node_vec1, node_vec2):
        B = x.size(0)  # (B, N, 1, nhid)
        dim = node_vec1.shape[-1]  # (N, 1, d)

        random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
        random_matrix = create_random_matrix(self.random_feature_dim, dim, seed=random_seed).to( node_vec1.device)  # (d, r)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)  # [B, N, 1, r]
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)  # [B, N, 1, r]

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)

        return x, node_vec1_prime, node_vec2_prime


class linearized_conv(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
        super(linearized_conv, self).__init__()

        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim

        self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.output_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_app_layer = conv_approximation(self.dropout, self.tau, self.random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        x = self.input_fc(input_data)
        x = self.activation(x) * self.output_fc(input_data)
        x = self.dropout_layer(x)

        x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)  # (B, dim*4, N, 1)

        return x, node_vec1_prime, node_vec2_prime
