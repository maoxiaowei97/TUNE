import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.argparser import ws
import time
from tqdm import tqdm

class Dataset_list(Dataset):
    def __init__(self, route_full, estimated_route_segment_travel_time_distribution, start_ts,
                 start_day, route_segment_num, total_duration, segment_travel_time_label,  traj_id, sub_traj_idx, duration_from_last_estimate):
        # 初始化所有字段
        self.route_full = route_full
        self.estimated_route_segment_travel_time_distribution = estimated_route_segment_travel_time_distribution
        self.start_ts = start_ts
        self.start_day = start_day
        self.route_segment_num = route_segment_num
        self.total_duration = total_duration
        self.segment_travel_time_label = segment_travel_time_label
        self.traj_id = traj_id
        self.sub_traj_idx = sub_traj_idx
        self.duration_from_last_estimate = duration_from_last_estimate

    def __len__(self):
        # 返回数据集的长度，假设所有字段长度相同
        return len(self.route_full)

    def __getitem__(self, idx):
        return (self.route_full[idx], self.estimated_route_segment_travel_time_distribution[idx], self.start_ts[idx],
                self.start_day[idx], self.route_segment_num[idx], self.total_duration[idx], self.segment_travel_time_label[idx],
                self.traj_id[idx], self.sub_traj_idx[idx],  self.duration_from_last_estimate[idx])


def collate_fn_list(batch):

    (route_full, route_segment_travel_time_distribution,
     start_ts, start_day, route_segment_num, total_duration, segment_travel_time_label, traj_id,
     sub_traj_idx, duration_from_last_estimate) = zip(*batch)

    route_full = torch.tensor(route_full).long()
    route_segment_travel_time_distribution = torch.tensor(route_segment_travel_time_distribution, dtype=torch.float32)

    start_ts = torch.tensor(start_ts, dtype=torch.float32)
    start_day = torch.tensor(start_day).long()
    route_segment_num = torch.tensor(route_segment_num).long()
    total_duration = torch.tensor(total_duration, dtype=torch.float32)
    segment_travel_time_label = torch.tensor(segment_travel_time_label, dtype=torch.float32)
    traj_id = torch.tensor(traj_id, dtype=torch.long)
    sub_traj_idx = torch.tensor(sub_traj_idx, dtype=torch.long)
    duration_from_last_estimate = torch.tensor(duration_from_last_estimate, dtype=torch.float32)


    return (route_full, route_segment_travel_time_distribution,\
            start_ts, start_day, route_segment_num,\
             total_duration, segment_travel_time_label, traj_id, sub_traj_idx, duration_from_last_estimate)

def dir_check(path):
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

class Trainer:
    def __init__(self, model: nn.Module, dataset, device, args):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.rho = args.rho
        self.early_stop = args.early_stop

    def train_all_eta(self, args):
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('local time: ', local_time)
        optimizer = torch.optim.Adam(self.model.parameters(), args.lr)

        pickle_file_path = ws + '/data/train_val_test_0505_sub.npz'
        data_to_load = np.load(pickle_file_path, allow_pickle=True)
        print('loaded train_val_test_held data: ', pickle_file_path)

        train_data = data_to_load['train'].item()
        val_data = data_to_load['val'].item()
        test_data = data_to_load['test'].item()


        def loss_fn(y_pred_mean, y_pred_lower, y_pred_upper, y_true):

            rho = 0.1
            loss0 = torch.abs(y_pred_mean - y_true)
            loss1 = torch.max(y_true - y_pred_upper, torch.tensor([0.]).cuda()) * 2 / rho
            loss2 = torch.max(y_pred_lower - y_true, torch.tensor([0.]).cuda()) * 2 / rho
            loss3 = torch.abs(y_pred_upper - y_pred_lower)
            loss = loss0  + loss1 + loss2  + loss3
            return loss.mean()


        def picp(y_pred_lower, y_pred_upper, y_true):
            picp = (((y_true < y_pred_upper.reshape(-1)) & (y_true > y_pred_lower.reshape(-1))) + 0).sum() / len(y_true)
            return picp


        traindataset = Dataset_list(train_data['route_full'], train_data['estimated_route_segment_travel_time_distribution'],
                                    train_data['start_ts'], train_data['start_day'], train_data['route_segment_num'],  train_data['total_duration'], train_data['segment_travel_time_label'],
                                    train_data['traj_id'], train_data['sub_traj_idx'],  train_data['duration_from_last_estimate'])

        traindataloader = DataLoader(traindataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list, drop_last=True)

        valdataset = Dataset_list(val_data['route_full'], val_data['estimated_route_segment_travel_time_distribution'],
                                    val_data['start_ts'], val_data['start_day'], val_data['route_segment_num'],  val_data['total_duration'], val_data['segment_travel_time_label'],
                                    val_data['traj_id'], val_data['sub_traj_idx'], val_data['duration_from_last_estimate'])

        valdataloader = DataLoader(valdataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list,  drop_last=True)

        testdataset = Dataset_list(test_data['route_full'], test_data['estimated_route_segment_travel_time_distribution'],
                                    test_data['start_ts'], test_data['start_day'], test_data['route_segment_num'],  test_data['total_duration'], test_data['segment_travel_time_label'],
                                    test_data['traj_id'], test_data['sub_traj_idx'],  test_data['duration_from_last_estimate'])

        testdataloader = DataLoader(testdataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list,  drop_last=True)

        train_loss = []
        early_stop = EarlyStop(mode='minimize', patience=self.early_stop)
        for epoch in range(args.n_epoch):
            if early_stop.stop_flag: break
            self.model.train()
            print('train epoch {}'.format(epoch))
            for batch in tqdm(traindataloader):
                (route_full,
                 route_segment_travel_time_distribution,
                 start_ts,
                 start_day,
                 route_segment_num,
                 total_duration, segment_travel_time_label, traj_id, sub_traj_idx, duration_from_last_estimate) = batch
                loss, predict_mean, valid_segment_mean, batch_covs, pred_lower, pred_upper = self.model(route_full, route_segment_travel_time_distribution, route_segment_num, start_ts,
                                                 segment_travel_time_label, total_duration, self.device)
                mis_loss = loss_fn(predict_mean.reshape(-1), pred_lower, pred_upper, total_duration.reshape(-1).float().to(self.device))
                loss = loss  + mis_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            print(f'training... loss of epoch: {epoch}: ' + str((sum(train_loss) / len(train_loss))))
            if epoch % 1 == 0:
                print(f'validation... of epoch {epoch}')
                predicts = []
                predicts_lower = []
                predicts_upper = []
                label = []
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(valdataloader):
                        (route_full,
                         route_segment_travel_time_distribution,
                         start_ts,
                         start_day,
                         route_segment_num,
                         total_duration, segment_travel_time_label, traj_id, sub_traj_idx, duration_from_last_estimate) = batch
                        loss, predict_mean, valid_segment_mean, batch_covs, pred_lower, pred_upper = self.model.test(route_full, route_segment_travel_time_distribution,
                                                        route_segment_num, start_ts,
                                                        segment_travel_time_label, total_duration, self.device)
                        predicts += predict_mean.reshape(-1).tolist()
                        predicts_lower += pred_lower.reshape(-1).tolist()
                        predicts_upper += pred_upper.reshape(-1).tolist()
                        label += total_duration.reshape(-1).float().tolist()

                    predicts = np.array(predicts).reshape(-1)
                    label = np.array(label).reshape(-1)
                    predicts_lower = np.array(predicts_lower).reshape(-1)
                    predicts_upper = np.array(predicts_upper).reshape(-1)
                    from sklearn.metrics import mean_squared_error as mse
                    from sklearn.metrics import mean_absolute_error as mae
                    def mape_(label, predicts):
                        return (abs(predicts - label) / label).mean()

                    val_mape = mape_(label, predicts)
                    val_mse = mse(label, predicts)
                    val_mae = mae(label, predicts)
                    val_width = predicts_upper - predicts_lower
                    val_picp = picp(predicts_lower, predicts_upper, label)

                    print('val point estimation: MAPE:%.3f\tRMSE:%.2f\tMAE:%.2f' % (val_mape * 100, np.sqrt(val_mse), val_mae))
                    print('val_width: ' + str(np.mean(val_width.reshape(-1))))
                    print('val_picp: ' + str(val_picp))
                is_best_change = early_stop.append(val_mape * 100)
                if is_best_change:
                    dir_check(ws + f"/model_params/{local_time}/")
                    best_model_path = ws + f"/model_params/{local_time}/finished_{epoch}.pth"
                    torch.save(self.model.state_dict(), best_model_path)
                    print('val best model saved at: ', best_model_path)
                    self.model.load_state_dict(torch.load(best_model_path))
                    print('val best model loaded')
        print('testing...')

        print('load best val model at: ', best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))
        print('testing, best val model loaded')
        predicts = []
        predicts_lower = []
        predicts_upper = []
        label = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(testdataloader):
                (route_full,
                 route_segment_travel_time_distribution,
                 start_ts,
                 start_day,
                 route_segment_num,
                 total_duration, segment_travel_time_label, traj_id, sub_traj_idx,
                 duration_from_last_estimate) = batch
                loss, predict_mean, valid_segment_mean, batch_covs, pred_lower, pred_upper = self.model.test(route_full,
                                                                                                        route_segment_travel_time_distribution,
                                                                                                        route_segment_num,
                                                                                                        start_ts,
                                                                                                        segment_travel_time_label,
                                                                                                        total_duration,
                                                                                                        self.device)
                predicts += predict_mean.reshape(-1).tolist()
                predicts_lower += pred_lower.reshape(-1).tolist()
                predicts_upper += pred_upper.reshape(-1).tolist()
                label += total_duration.reshape(-1).float().tolist()

            predicts = np.array(predicts).reshape(-1)
            label = np.array(label).reshape(-1)
            predicts_lower = np.array(predicts_lower).reshape(-1)
            predicts_upper = np.array(predicts_upper).reshape(-1)
            from sklearn.metrics import mean_squared_error as mse
            from sklearn.metrics import mean_absolute_error as mae
            def mape_(label, predicts):
                return (abs(predicts - label) / label).mean()

            val_mape = mape_(label, predicts)
            val_mse = mse(label, predicts)
            val_mae = mae(label, predicts)
            val_width = predicts_upper - predicts_lower
            val_picp = picp(predicts_lower, predicts_upper, label)

            print('test point estimation: MAPE:%.3f\tRMSE:%.2f\tMAE:%.2f' % (val_mape * 100, np.sqrt(val_mse), val_mae))
            print('test_width: ' + str(np.mean(val_width.reshape(-1))))
            print('test_picp: ' + str(val_picp))


class EarlyStop():
    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1  # the best epoch
        self.is_best_change = False  # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(
            min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch  # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]


def whether_stop(metric_lst=[], n=2, mode='maximize'):
    if len(metric_lst) < 1: return False  # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n