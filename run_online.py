import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from utils.argparser import ws
import time
import os
from data.dataset import TrajFastDataset
from utils.argparser import get_argparser
from datetime import datetime
import torch.distributions as dist
import multiprocessing as mp


def dir_check(path):
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

def worker_init_fn():
    torch.cuda.init()

class Online_inference:
    def __init__(self, model: nn.Module, dataset, device, args):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.rho = args.rho
        self.early_stop = args.early_stop
        if os.path.exists(args.best_model_path):
            self.model.load_state_dict(torch.load(args.best_model_path))
        else:
            print(f"Model file not found: {args.best_model_path}.")

    def process_trajectory(self, args):
        """
        args: (traj_id, group_df, beta, z_90)
        Returns: dict of {traj_id, sub_idx, picp, iw, re_estimate, last_segment_condition,current_segment_condition, lower_bound, upper_bound, estimated_mean_travel_time, total_remain_time) }
        """
        traj_id, df, beta, z_90 = args
        df = df.sort_values(by='sub_traj_idx')
        results = {}

        last_remain_segment_num = None
        last_pred_segment_for_remaining_route = None
        last_predict_remain_total_time = None
        last_predict_segments_mean = None
        last_predict_segments_cov = None

        for idx, row in df.iterrows():
            sub_idx = row['sub_traj_idx']
            if sub_idx == 1:
                remain_seg_num = row['route_segment_num']
                remaining_total_segment_num = row['route_segment_num']
                last_remain_segment_num = row['route_segment_num']
                """
                input of route travel time predictor
                """
                route_full = torch.stack([torch.LongTensor(row['route_full'])])
                route_segment_travel_time_mean = np.array(row['estimated_route_segment_travel_time_distribution'])[:, -1][:remain_seg_num]
                route_segment_num = row['route_segment_num']
                start_ts = torch.LongTensor([row['start_ts']])
                segment_travel_time_label = torch.stack([torch.FloatTensor(row['segment_travel_time_label'])])
                total_duration = torch.LongTensor([row['total_duration']])
                last_predict_remain_total_time = row['total_duration']
                route_segment_travel_time_distribution = torch.stack([torch.FloatTensor(row['estimated_route_segment_travel_time_distribution'])])
                route_segment_num = torch.LongTensor([route_segment_num])

                """
                segment_pred_mean, cov, estimated_segment_mean, 
                """
                _, predict_mean, pred_segment_mean, cov, _, _ = self.model.test(route_full,   route_segment_travel_time_distribution,
                                                                                                             route_segment_num,
                                                                                                             start_ts,
                                                                                                             segment_travel_time_label,
                                                                                                             total_duration,
                                                                                                             self.device)
                last_pred_segment_for_remaining_route = pred_segment_mean.squeeze(0).squeeze(-1)
                last_predict_segments_mean = last_pred_segment_for_remaining_route[:remain_seg_num]
                last_predict_segments_cov = cov[0]

                last_segment_condition_at_departure = route_segment_travel_time_mean
                last_segment_condition = np.sum(route_segment_travel_time_mean)
                current_segment_condition = np.sum(route_segment_travel_time_mean)

                # Compute initial intervals
                Cov_1 = last_predict_segments_cov[:remain_seg_num, :remain_seg_num]
                w_1 = torch.ones(remain_seg_num, dtype=torch.float32).to(self.device)
                mean_sum_1 = w_1 @ last_predict_segments_mean
                var_sum_1 = w_1 @ Cov_1 @ w_1
                std_sum_1 = torch.sqrt(var_sum_1)
                lower_90_1 = int((mean_sum_1 - z_90 * std_sum_1).item())
                upper_90_1 = int((mean_sum_1 + z_90 * std_sum_1).item())
                iw_1 = upper_90_1 - lower_90_1
                real_remaining_time = row['total_duration']
                picp_1 = 1 if (lower_90_1 <= real_remaining_time <= upper_90_1) else 0
                results[idx] = ( traj_id, sub_idx, picp_1, iw_1, 0.0, last_segment_condition, current_segment_condition, lower_90_1, upper_90_1, int(mean_sum_1.item()), real_remaining_time)

            else:
                # sub_idx > 1, continue from previous
                current_route_seg_num = row['route_segment_num']
                current_segment_condition = np.array(row['estimated_route_segment_travel_time_distribution'])[:, -1][:current_route_seg_num]
                traveled_seg_num_from_last = last_remain_segment_num - current_route_seg_num

                # Summation of previously predicted times for traveled segments
                last_pred_for_traveled_segment = torch.sum(last_pred_segment_for_remaining_route[:traveled_seg_num_from_last]).item()
                # Real traveled time from the last estimate
                traveled_time = (last_predict_remain_total_time - row['total_duration'])
                traveled_route_mape = (abs(last_pred_for_traveled_segment - traveled_time) / traveled_time * 100  if traveled_time != 0 else 0)
                remaining_route_estimated_total = np.sum(current_segment_condition)
                """
                last estimated time for remaining route
                """
                mean_future = last_predict_segments_mean[traveled_seg_num_from_last: remaining_total_segment_num]
                cov_indices_future = np.arange(traveled_seg_num_from_last, remaining_total_segment_num)
                Cov_future = last_predict_segments_cov[cov_indices_future][:, cov_indices_future]
                w_future = torch.ones(len(cov_indices_future), dtype=torch.float32).to(self.device)
                mean_sum_future = (w_future @ mean_future)
                var_sum_future = w_future @ Cov_future @ w_future
                std_sum_future = torch.sqrt(var_sum_future)

                lower_future = int((mean_sum_future - z_90 * std_sum_future).item())
                upper_future = int((mean_sum_future + z_90 * std_sum_future).item())

                cover_remain_route_time = 1 if (lower_future <= remaining_route_estimated_total <= upper_future) else 0

                if (traveled_route_mape >= beta)  or (cover_remain_route_time == 0):
                    """
                    重新估计，需要更新信息，包括last_segment_condition
                    """
                    re_estimate_val = 1.0
                    remaining_total_segment_num = row['route_segment_num']
                    last_predict_remain_total_time = row['total_duration']
                    last_remain_segment_num = row['route_segment_num']

                    """
                    input of route travel time predictor
                    """
                    route_full = torch.stack([torch.LongTensor(row['route_full'])])
                    route_segment_num = row['route_segment_num']
                    start_ts = torch.LongTensor([row['start_ts']])
                    segment_travel_time_label = torch.stack([torch.FloatTensor(row['segment_travel_time_label'])])
                    total_duration = torch.LongTensor([row['total_duration']])
                    route_segment_travel_time_distribution = torch.stack([torch.FloatTensor(row['estimated_route_segment_travel_time_distribution'])])
                    route_segment_num = torch.LongTensor([route_segment_num])
                    """
                    segment_pred_mean, cov, estimated_segment_mean, 
                    """
                    _, predict_mean, pred_segment_mean, cov, _, _ = self.model.test(route_full,
                                                                                    route_segment_travel_time_distribution,
                                                                                    route_segment_num,
                                                                                    start_ts,
                                                                                    segment_travel_time_label,
                                                                                    total_duration,
                                                                                    self.device)
                    last_pred_segment_for_remaining_route = pred_segment_mean.squeeze(0).squeeze(-1)

                    last_predict_segments_mean = last_pred_segment_for_remaining_route[:last_remain_segment_num]
                    last_predict_segments_cov = cov[0]

                    mean_current = last_predict_segments_mean
                    cov_indices_current = np.arange(0, last_remain_segment_num)

                    # Compute initial intervals
                    Cov_b_current = last_predict_segments_cov[cov_indices_current][:, cov_indices_current]
                    w_current = torch.ones(len(cov_indices_current), dtype=torch.float32).to(self.device)
                    mean_sum_current = w_current @ mean_current
                    var_sum_current = w_current @ Cov_b_current @ w_current
                    std_sum_current = torch.sqrt(var_sum_current)
                    lower_90_current = int((mean_sum_current - z_90 * std_sum_current).item())
                    upper_90_current = int((mean_sum_current + z_90 * std_sum_current).item())
                    iw_current = upper_90_current - lower_90_current
                    real_rem_time = row['total_duration']
                    picp_current = 1 if (lower_90_current <= real_rem_time <= upper_90_current) else 0
                    last_segment_condition = np.sum( last_segment_condition_at_departure[-current_route_seg_num:])  # 出发时观测的剩余路径通行时间，还剩多少路段取多少个
                    results[idx] = (traj_id, sub_idx, picp_current, iw_current, re_estimate_val, last_segment_condition,remaining_route_estimated_total, lower_90_current, upper_90_current, int(mean_sum_current.item()), real_rem_time)

                else:
                    re_estimate_val = 0.0
                    remain_time = row['total_duration']
                    mean_current = last_predict_segments_mean[traveled_seg_num_from_last: last_remain_segment_num]
                    cov_sub_indices_current = np.arange(traveled_seg_num_from_last, last_remain_segment_num)
                    Cov_b_current = last_predict_segments_cov[cov_sub_indices_current][:, cov_sub_indices_current]
                    w_current = torch.ones(len(cov_sub_indices_current), dtype=torch.float32).to(self.device)
                    mean_sum_current = w_current @ mean_current
                    var_sum_current = w_current @ Cov_b_current @ w_current
                    std_sum_current = torch.sqrt(var_sum_current)
                    lower_90_current = int((mean_sum_current - z_90 * std_sum_current).item())
                    upper_90_current = int((mean_sum_current + z_90 * std_sum_current).item())
                    iw_current = upper_90_current - lower_90_current
                    picp_current = 1 if (lower_90_current <= remain_time <= upper_90_current) else 0
                    last_segment_condition = np.sum(last_segment_condition_at_departure[-current_route_seg_num:])
                    results[idx] = (traj_id, sub_idx, picp_current, iw_current, re_estimate_val, last_segment_condition, remaining_route_estimated_total, lower_90_current, upper_90_current, int(mean_sum_current.item()), remain_time)

        return results


    def run_online(self):
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('local time: ', local_time)

        pickle_file_path = ws + '/data/train_val_test_0505_sub.npz'
        data_to_load = np.load(pickle_file_path, allow_pickle=True)
        print('loaded train_val_test_held data: ', pickle_file_path)
        test_data = data_to_load['test'].item()

        df = pd.DataFrame({ 'traj_id': test_data['traj_id'][3:],
        'sub_traj_idx': test_data['sub_traj_idx'][3:],
        'total_duration': test_data['total_duration'][3:],
        'segment_travel_time_label': test_data['segment_travel_time_label'][3:],
        'start_ts': test_data['start_ts'][3:],
        'start_day': test_data['start_day'][3:],
        'route_segment_num': test_data['route_segment_num'][3:],
        'route_full':  [list(x) for x in test_data['route_full'][3:]],
        'estimated_route_segment_travel_time_distribution': [list(x) for x in test_data['estimated_route_segment_travel_time_distribution'][3:]]
        })
        grouped = df.groupby('traj_id')

        traj_id_all = np.zeros(len(df))
        sub_idx_all = np.zeros(len(df))
        picp_all = np.zeros(len(df))
        iw_all = np.zeros(len(df))
        re_estimate_all = np.zeros(len(df))
        last_segment_condition_all = np.zeros(len(df))
        current_segment_condition_all = np.zeros(len(df))
        lower_all = np.zeros(len(df))
        upper_all = np.zeros(len(df))
        mean_all = np.zeros(len(df))
        true_time_all = np.zeros(len(df))

        beta = 0.23
        num_workers = 5
        z_90 = dist.Normal(0, 1).icdf(torch.tensor([1.0 - (1.0 - 0.9) / 2]))[0].item()
        tasks = []
        mp.set_start_method('spawn', force=True)  # Use spawn start method for CUDA compatibility

        for traj_id, group in grouped:
            tasks.append((traj_id, group, beta, z_90))

        # result = self.process_trajectory(tasks[1])
        # c = result
        with mp.Pool(processes=num_workers, initializer=worker_init_fn) as pool:
            results_iter = pool.map(self.process_trajectory, tasks)

        for res_dict in results_iter:
            for idx, (traj_id, sub_idx, picp_val, iw_val, re_est_val, last_segment_condition, current_segment_condition, lower, upper, mean, remain_time) in res_dict.items():
                traj_id_all[idx] = traj_id
                sub_idx_all[idx] = sub_idx
                picp_all[idx] = picp_val
                iw_all[idx]   = iw_val
                re_estimate_all[idx] = re_est_val
                last_segment_condition_all[idx] = last_segment_condition
                current_segment_condition_all[idx] = current_segment_condition
                lower_all[idx] = lower
                upper_all[idx] = upper
                mean_all[idx] = mean
                true_time_all[idx] = remain_time

        result_df = pd.DataFrame({
            'traj_id': traj_id_all,
            'sub_idx': sub_idx_all,
            'picp': picp_all,
            'iw': iw_all,
            're_estimate': re_estimate_all,
            'last_segment_condition': last_segment_condition_all,
            'current_segment_condition': current_segment_condition_all,
            'lower': lower_all,
            'upper': upper_all,
            'mean': mean_all,
            'true_time': true_time_all
        })


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    parser = get_argparser()
    args = parser.parse_args()
    save_time = f'checkpoint_t{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

    if args.device == "default":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(device)

    dataset = TrajFastDataset(args.d_name, args.path, device, is_pretrain=False)
    n_vertex = dataset.n_vertex
    print(f"vertex: {n_vertex}")

    print(args)

    from uncertainty_quantification.multi_gaussian import MGUQ_network

    model = MGUQ_network(args).to(device)
    inference_process = Online_inference(model, dataset, device, args)
    inference_process.run_online()