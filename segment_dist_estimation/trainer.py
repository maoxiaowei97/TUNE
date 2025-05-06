import torch
import metrics
from model import LGDE
import time
class Trainer():
    def __init__(self, args, device):
        self.model = LGDE(args).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=0.1, verbose=False)
        
        self.loss = metrics.masked_mae
        self.use_spatial = args.use_spatial
        self.grad_clip = args.grad_clip

    def train(self, input, input_d_ts, label_dist, device, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        start_time = time.time()
        loss_mae, loss_kl, loss_mape, loss_rmse, predicted_probs, predicted_mean = self.model(input,  input_d_ts, label_dist, device)
        end_time = time.time()

        # Calculate and print the time taken for this batch
        batch_time = end_time - start_time
        print(f"Batch processed in {batch_time:.4f} seconds in training")


        loss = loss_mae + loss_kl + loss_mape


        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item(), loss_mape.item(), loss_mae.item(), loss_kl.item(), loss_rmse.item()

    def eval(self, input, input_d_ts, label_dist, device):
        self.model.eval()
        start_time = time.time()
        loss_mae, loss_kl, loss_mape, loss_rmse, predicted_probs, predicted_mean= self.model(input, input_d_ts, label_dist, device)
        end_time = time.time()
        batch_time = end_time - start_time
        print(f"Batch processed in {batch_time:.4f} seconds in testing")
        loss = loss_mae + loss_kl + loss_mape
        return loss.item(), loss_mape.item(), loss_mae.item(), loss_kl.item(), loss_rmse.item()

    def eval_save(self, input,  input_d_ts, label_dist, device):
        self.model.eval()
        loss_mae, loss_kl, loss_mape, loss_rmse, predicted_probs, predicted_mean = self.model(input, input_d_ts, label_dist, device)
        loss = loss_mae + loss_kl + loss_mape
        return loss.item(), loss_mape.item(), loss_mae.item(), loss_kl.item(), loss_rmse.item()
