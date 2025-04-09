import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:2", help="gpu device")
    # parser.add_argument("--seed", type=int, default=183, help="Random seed(ADNI)")
    parser.add_argument("--seed", type=int, default=183, help="Random seed(PD)")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=3e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=500, help='early stopping param')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size.')

    parser.add_argument('--cross_loss', type=float, default=1.0)
    parser.add_argument('--graph_loss', type=float, default=0.1)
    parser.add_argument('--time_loss', type=float, default=0.1)

    parser.add_argument("--dataset", type=str, default="PD", help="dataset: ADNI, PD, ABIDE")
    parser.add_argument("--mission", type=str, default="0_12", help="0_12, 0_1, 0_2, 1_2")

    # 消融实验
    parser.add_argument("--if_graph", type=bool, default=True, help="if_learing_graph_structure")
    parser.add_argument("--if_time_graph", type=bool, default=True, help="if_learing_time_graph")
    parser.add_argument("--if_time_mapper", type=bool, default=True, help="if_time_mapper")

    # 保存adj
    parser.add_argument("--save_adj", type=bool, default=False, help="save the adj")
    # t-SNE散点图
    parser.add_argument("--save_tSNE", type=bool, default=False, help="save the t-SNE embedding")

    # 数据集路径
    args = parser.parse_args()
    if args.dataset == 'ADNI':
        # parser.add_argument("--datapath", type=str, default="E:\Dataset\ADNI\ADNI_NC_SMC_EMCI_New.mat")
        parser.add_argument("--datapath", type=str, default="/lab/2023/yn/Dataset/fMRI/ADNI/ADNI.mat")
        parser.add_argument("--num_class", type=int, default=2)
        parser.add_argument("--num_node", type=int, default=90)
        parser.add_argument("--time_length", type=int, default=195)
        parser.add_argument("--window_num", type=int, default=6)
        parser.add_argument("--gcn_output", type=int, default=30)
        parser.add_argument("--windows_input", type=int, default=32)
        parser.add_argument("--windows_output", type=int, default=16)
        parser.add_argument("--window_size", type=int, default=0)
    elif args.dataset == 'PD':
        # parser.add_argument("--datapath", type=str, default="D:\YuanNing\Dataset\pd\PD_dataset.mat")
        parser.add_argument("--datapath", type=str, default="/lab/2023/yn/Dataset/fMRI/PD/PD_dataset.mat")
        parser.add_argument("--num_class", type=int, default=2)
        parser.add_argument("--num_node", type=int, default=116)
        parser.add_argument("--time_length", type=int, default=220)
        parser.add_argument("--window_num", type=int, default=4)
        parser.add_argument("--gcn_output", type=int, default=30)
        parser.add_argument("--windows_input", type=int, default=32)
        parser.add_argument("--windows_output", type=int, default=16)
        parser.add_argument("--window_size", type=int, default=55)
    elif args.dataset == 'ABIDE':
        # parser.add_argument("--datapath", type=str, default="D:\YuanNing\Dataset\ABIDE\ABIDE_fMRI.mat")
        parser.add_argument("--datapath", type=str, default="/lab/2023/yn/Dataset/ABIDE/ABIDE_fMRI.mat")
        parser.add_argument("--num_class", type=int, default=2)
        parser.add_argument("--num_node", type=int, default=116)
        parser.add_argument("--time_length", type=int, default=176)
        parser.add_argument("--window_num", type=int, default=3)
        parser.add_argument("--gcn_output", type=int, default=30)
        parser.add_argument("--windows_input", type=int, default=32)
        parser.add_argument("--windows_output", type=int, default=16)
        parser.add_argument("--window_size", type=int, default=55)
    else:
        raise ValueError("Wrong dataset!")

    return parser.parse_args()