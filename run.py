import torch
from data.dataset import TrajFastDataset
from utils.argparser import get_argparser
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.environ['CRYPTOGRAPHY_OPENSSL_NO_LEGACY'] = '4'

def dir_check(path):

    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

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

    if args.method == 'MGUQ':
        from uncertainty_quantification.trainer import Trainer
        from uncertainty_quantification.multi_gaussian import MGUQ_network
        print('method: ', args.method)

        model = MGUQ_network(args).to(device)
        trainer = Trainer(model, dataset, device, args)
        trainer.train_all_eta(args)


