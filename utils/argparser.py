import argparse
import os

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

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="configure all the settings")

    # device config
    parser.add_argument("-device", type=str, help="device, [cpu, cuda]", default="default")

    # path config
    parser.add_argument("-path", type=str, help="data path", default=ws + "/data/map")
    parser.add_argument("-model_path", type=str, help="model path", default=ws + "/model_params")
    parser.add_argument("-best_model_path", type=str, help="model path", default=ws + "/model_params/trained_model.pth")
    parser.add_argument("-res_path", type=str, help="results path", default=ws + "/results")

    # data config
    parser.add_argument("-d_name", type=str, default="chengdu")
    # MGUQ config
    parser.add_argument("-model_name", type=str, help="model name", default="MGUQ")
    parser.add_argument("-method", type=str, help="method: cold, naive", default="MGUQ")
    parser.add_argument("-x_emb_dim", type=int, help="vertex embedding dim", default=100)
    parser.add_argument("-E_U", type=int, help="embedding dim in MoEUQ", default=256)
    parser.add_argument("-C", type=int, help="number of experts", default=8)
    parser.add_argument("-k", type=int, help="number of selected experts", default=4)
    parser.add_argument("-L_T", type=int, help="number of layers for transformer in path prediction", default=1)
    parser.add_argument("-m", type=int, help="number of statistical travel time", default=5)
    parser.add_argument("-beta", type=int, help="threshold of mape", default=0.23)
    parser.add_argument("-c_f", type=int, help="confidence level of uncertainty quantification", default=0.9)
    parser.add_argument("-n_groups", type=int, help="number of groups for group normalization", default=8)
    parser.add_argument("-dropout", type=int, help="number of statistical travel time", default=0.05)
    parser.add_argument("-n_samples", type=int, help="number of statistical travel time", default=10)
    parser.add_argument("-id_embed_dim", type=int, help="embed dim of discrete features", default=20)
    parser.add_argument("-slice_dim", type=int, help="dim of time slices", default=721)
    parser.add_argument("-slice_embed_dim", type=int, help="dim of time slices", default=721)
    parser.add_argument("-segment_dim", type=int, help="dim of time slices", default=12693)
    parser.add_argument("-dist_dim", type=int, help="dim of distribution", default=36)

    # gp
    parser.add_argument("-mape_loss_weight", type=int, default=10)
    parser.add_argument("-gp_loss_weight", type=int, default=0.2)
    parser.add_argument("-mean_weight", type=int, default=0.9)
    parser.add_argument("-sample_weight", type=int, default=0.1)


    # confidence level
    parser.add_argument("-rho", type=int, default=0.1)
    parser.add_argument("-alpha", type=int, default=0.1)
    parser.add_argument("-delta", type=int, default=0.1)

    # training config
    parser.add_argument("-n_epoch", type=int, help="number of epoch", default=90)
    parser.add_argument("-bs", type=int, help="batch size", default=32)
    parser.add_argument("-lr", type=float, help="learning rate", default=0.001)



    # eval config
    parser.add_argument("-early_stop", type=int, help="patience of early stop", default=10)

    return parser
