import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()

class Dataset_list(Dataset):
    def __init__(self, train_input, train_input_d_ts, train_label_dist):
        self.train_input = train_input
        self.train_input_d_ts = train_input_d_ts
        self.train_label_dist = train_label_dist

    def __len__(self):
        # 返回数据集的长度，假设所有字段长度相同
        return len(self.train_input_d_ts)

    def __getitem__(self, idx):
        return (self.train_input[idx],  self.train_input_d_ts[idx]  , self.train_label_dist[idx])

def collate_fn_list(batch):

    (train_input, train_input_d_ts, train_label_dist) = zip(*batch)

    train_input = torch.tensor(np.stack(train_input),  dtype=torch.float32)
    train_input_d_ts = torch.tensor(np.stack(train_input_d_ts)).long()
    train_label_dist = torch.tensor(np.stack(train_label_dist), dtype=torch.float32)

    return train_input,  train_input_d_ts, train_label_dist

def load_dataset(args):

    train_data = np.load(ws + '/segment_distribution_dataset/train_sub_true.npz', allow_pickle=True)
    val_data = np.load(ws + '/segment_distribution_dataset/val_sub_true.npz', allow_pickle=True)
    test_data = np.load(ws + '/segment_distribution_dataset/test_sub_true.npz', allow_pickle=True)

    traindataset = Dataset_list(train_data['train_input'], train_data['train_input_d_ts'],  train_data['train_label_dist'])

    traindataloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_list, drop_last=False)

    valdataset = Dataset_list(val_data['train_input'],  val_data['train_input_d_ts'],  val_data['train_label_dist'])

    valdataloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_list, drop_last=False)

    testdataset = Dataset_list(test_data['train_input'], test_data['train_input_d_ts'],  test_data['train_label_dist'])

    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_list, drop_last=False)


    return traindataloader, valdataloader, testdataloader
