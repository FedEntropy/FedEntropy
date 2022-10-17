from torchvision import datasets, transforms
from Utils import mydata
from Utils.sampling import *


def get_dataset(args):
    if args.dataset == "cifar10":
        print("Dataset: cifar10")
        trans_cifar10_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        trans_cifar10_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_test)
        print("训练集的图像数量为: ", len(dataset_train))
        print("测试集的图像数量为: ", len(dataset_test))

        if args.rule == "iid":
            pass
        elif args.rule == "ill":
            dict_users = cifar10_noniid(dataset_train, args)
        elif args.rule == "Drichlet":
            min_size = 0
            min_require_size = 10
            K = args.num_classes
            y_train = np.array(dataset_train.targets)
            N = len(dataset_train)
            dict_users = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(args.num_users)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(args.Drichlet_arg, args.num_users))
                    proportions = np.array(
                        [p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(args.num_users):
                # np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        return dataset_train, dataset_test, dict_users

    elif args.dataset == "cifar100":
        print("Dataset: cifar100")
        trans_cifar100 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = mydata.CIFAR100_coarse('data/cifar100_coarse', train=True, download=True,
                                               transform=trans_cifar100)
        dataset_test = mydata.CIFAR100_coarse('data/cifar100_coarse', train=False, download=True,
                                              transform=trans_cifar100)
        print("训练集的图像数量为: ", len(dataset_train))
        print("测试集的图像数量为: ", len(dataset_test))

        if args.rule == "iid":
            pass
        elif args.rule == "ill":
            dict_users = cifar100_noniid(dataset_train, args)
        elif args.rule == "Drichlet":
            min_size = 0
            min_require_size = 10
            K = args.num_classes
            y_train = np.array(dataset_train.targets)
            N = len(dataset_train)
            dict_users = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(args.num_users)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(args.Drichlet_arg, args.num_users))
                    proportions = np.array(
                        [p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(args.num_users):
                # np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        return dataset_train, dataset_test, dict_users

    elif args.dataset == 'cinic10':
        trans_cinic10 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205756, 0.23828046, 0.25874835])])
        dataset_train = datasets.ImageFolder('data/cinic10/train', transform=trans_cinic10)
        dataset_test = datasets.ImageFolder('data/cinic10/test', transform=trans_cinic10)
        print("训练集的图像数量为: ", len(dataset_train))
        print("测试集的图像数量为: ", len(dataset_test))

        if args.rule == "iid":
            pass
        elif args.rule == "ill":
            dict_users = cinic10_noniid(dataset_train, args)
        elif args.rule == "Drichlet":
            min_size = 0
            min_require_size = 10
            K = args.num_classes
            y_train = np.array(dataset_train.targets)
            N = len(dataset_train)
            dict_users = {}

            while min_size < min_require_size:
                idx_batch = [[] for _ in range(args.num_users)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(args.Drichlet_arg, args.num_users))
                    proportions = np.array(
                        [p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(args.num_users):
                # np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        return dataset_train, dataset_test, dict_users

    else:
        print("unrecognized dataset")

