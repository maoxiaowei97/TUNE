import torch
import numpy as np
import argparse
import time
import util
import copy
from trainer import Trainer
from tqdm import tqdm
import os
from util import ws
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data_path',type=str,default=ws + '/segment_distribution_dataset',help='data path')
parser.add_argument('--hid_dim',type=int,default=256,help='')
parser.add_argument('--in_dim',type=int,default=10,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=12040,help='number of nodes')
parser.add_argument('--num_layers',type=int,default=1,help='number of layers')
parser.add_argument('--tau',type=int,default=0.25,help='temperature coefficient')
parser.add_argument('--random_feature_dim',type=int,default=64,help='random feature dimension')
parser.add_argument('--node_dim',type=int,default=256,help='node embedding dimension')
parser.add_argument('--time_dim',type=int,default=32,help='time embedding dimension')
parser.add_argument('--time_num',type=int,default=721,help='time in day')
parser.add_argument('--input_emb_dim',type=int,default=55,help='embed dim of input dist')
parser.add_argument('--dist_dim',type=int,default=35,help='dim of dist')
parser.add_argument('--week_num',type=int,default=7,help='day in week')
parser.add_argument('--use_residual', default=True, help='use residual connection')
parser.add_argument('--use_bn', default=True, help='use batch normalization')
parser.add_argument('--use_spatial', default=True, help='use spatial loss')
parser.add_argument('--use_long', default=False, help='use long-term preprocessed features')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--grad_clip',type=float,default=5,help='gradient cliip')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--milestones',type=list,default=[80, 100],help='optimizer milestones')
parser.add_argument('--patience',type=int,default=30,help='early stopping')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--save',type=str,default=ws + '/segment_dist_estimation/checkpoint',help='save path')
parser.add_argument('--checkpoint',type=str,default=ws + '/segment_dist_estimation/checkpoint/LGDE.pth',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
args = parser.parse_args()
print(args)
IS_TEST = False
if IS_TEST:
    args.num_nodes = 100

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('local time: ', local_time)
    args.device = torch.device(args.device)

    train_dataloader, val_dataloader, test_dataloader = util.load_dataset(args)

    trainer = Trainer(args, device)

    print("start training...",flush=True)
    his_loss =[]
    test_time = []
    val_time = []
    train_time = []

    wait = 0
    min_val_mape = np.inf
    
    for i in range(1, args.epochs+1):
        # train
        train_loss = []
        train_mape = []
        train_mae = []
        train_kl = []
        train_rmse = []
        t1 = time.time()

        for batch in tqdm(train_dataloader):
            (train_input, train_input_d_ts, train_label_dist) = batch

            metrics = trainer.train(train_input, train_input_d_ts, train_label_dist, device, i)

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_mae.append(metrics[2])
            train_kl.append(metrics[3])
            train_rmse.append(metrics[4])
            t2 = time.time()
            train_time.append(t2-t1)
            
        if i % args.print_every == 0:

            log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train MAE: {:.4f}, Train KL: {:.4f}, Train RMSE: {:.4f}'.format(
                i, np.mean(train_loss), np.mean(train_mape), np.mean(train_mae), np.mean(train_kl), np.mean(train_rmse))
            print(log, flush=True)

        trainer.scheduler.step()
        
        # validation
        valid_mape = []
        valid_mae = []
        valid_kl = []
        valid_rmse = []

        s1 = time.time()
        for batch in tqdm(val_dataloader):
            (train_input, train_input_d_ts, train_label_dist) = batch

            metrics = trainer.eval(train_input, train_input_d_ts, train_label_dist, device)

            valid_mape.append(metrics[1])
            valid_mae.append(metrics[2])
            valid_kl.append(metrics[3])
            valid_rmse.append(metrics[4])
        s2 = time.time()
        log = 'Epoch: {:03d}, Validation Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_mae = np.mean(train_mae)
        mtrain_kl = np.mean(train_kl)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_mape = np.mean(valid_mape)
        mvalid_mae = np.mean(valid_mae)
        mvalid_kl = np.mean(valid_kl)
        mvalid_rmse = np.mean(valid_rmse)

        if mvalid_mape < min_val_mape:
            wait = 0
            min_val_mape = mvalid_mape
            best_epoch = i
            print('best epoch: {:04d}'.format(best_epoch))
            best_state_dict = copy.deepcopy(trainer.model.state_dict())
            dir_check(ws + f"/segment_dist_estimation/checkpoint/{local_time}/")
            best_model_path =  ws + f"/segment_dist_estimation/checkpoint/{local_time}/finished_{best_epoch}.pth"
            torch.save(trainer.model.state_dict(), best_model_path)
            trainer.model.load_state_dict(torch.load(best_model_path))
            print('val best model loaded')
        else:
            wait += 1
            if wait >= args.patience:
                break
        log = 'Train Loss: {:.4f}, Train MAPE: {:.4f}, Train MAE: {:.4f}, Train KL: {:.4f}, Train RMSE: {:.4f},  Valid MAPE: {:.4f}, Valid MAE: {:.4f}, Valid KL: {:.4f}, Valid RMSE: {:.4f}'
        print(log.format(mtrain_loss, mtrain_mape, mtrain_mae, mtrain_kl, mtrain_rmse, mvalid_mape, mvalid_mae, mvalid_kl, mvalid_rmse), flush=True)

    # test
    trainer.model.load_state_dict(best_state_dict)
    s1 = time.time()
    test_loss = []
    test_mape = []
    test_mae = []
    test_kl = []
    test_rmse = []

    for batch in tqdm(test_dataloader):
        (train_input, train_input_d_ts, train_label_dist) = batch

        metrics = trainer.eval(train_input, train_input_d_ts, train_label_dist, device)

        test_loss.append(metrics[0])
        test_mape.append(metrics[1])
        test_mae.append(metrics[2])
        test_kl.append(metrics[3])
        test_rmse.append(metrics[4])

    mtest_mape = np.mean(test_mape)
    mtest_mae = np.mean(test_mae)
    mtest_kl = np.mean(test_kl)
    mtest_rmse = np.mean(test_rmse)


    s2 = time.time()
    log = 'Epoch: {:03d}, Test Inference Time: {:.4f} secs'
    print(log.format(i,(s2-s1)))
    test_time.append(s2-s1)

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f},  Test KL: {:.4f},  Test RMSE: {:.4f}'
    print(log.format(mtest_mae,mtest_mape, mtest_kl, mtest_rmse), flush=True)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))




if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
