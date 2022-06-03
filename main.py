from utils.options import args_parser
from utils.Get_Dataset import get_dataset
from models.Nets import *
from Algorithm.DynamicEntropy import *
import numpy as np


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    dataset_train, dataset_test, dict_users = get_dataset(args)

    # np.save('cinic10_sample5_0.1.npy', dict_users)
    dict_users = np.load('cifar10_sample1.npy', allow_pickle=True).item()

    if args.dataset == "mnist":
        net_glob = CNNFashionMnist(args).to(args.device)
    elif args.dataset == "fashion-mnist":
        net_glob = CNNFashionMnist(args).to(args.device)
    elif args.dataset == "femnist":
        net_glob = CNNFashionMnist(args).to(args.device)
    elif args.dataset == "cifar10" or args.dataset == "cifar100":
        net_glob = CNNCifar(args).to(args.device)
    else:
        net_glob = CNNCifar(args).to(args.device)
    print(net_glob)

    DynamicEntropy(args, net_glob, dataset_train, dataset_test, dict_users)

