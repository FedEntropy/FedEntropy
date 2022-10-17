from Utils.Options import args_parserfrom Utils.GetData import get_datasetfrom Utils.GetModel import get_modelfrom Utils.SetRandomSeed import set_random_seedfrom Algorithm.algorithm import *import torchif __name__ == "__main__":    args = args_parser()    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')    set_random_seed(2022)    train_dataset, test_dataset, dict_users = get_dataset(args)    net_glob = get_model(args)    print(net_glob)    if args.algorithm == "FedAvg":        FedAvg(args, net_glob, train_dataset, test_dataset, dict_users)    if args.algorithm == "FedProx":        FedProx(args, net_glob, train_dataset, test_dataset, dict_users)    if args.algorithm == "moon":        moon(args, net_glob, train_dataset, test_dataset, dict_users)    if args.algorithm == "FedEntropy":        FedEntropy(args, net_glob, train_dataset, test_dataset, dict_users)