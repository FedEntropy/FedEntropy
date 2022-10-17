import numpy as np


def iid(dataset, args):
    num_items = int(len(dataset) / args.num_users)
    dict_users, all_idx = {}, [i for i in range(len(dataset))]
    for i in range(args.num_users):
        dict_users[i] = set(np.random.choice(all_idx, num_items, replace=False))
        all_idx = list(set(all_idx) - dict_users[i])
    return dict_users


def non_iid(dataset, args, num_shards, num_imgs):
    if args.ill_case == 1:
        return noniid_ratio_r_label_1(dataset, args.num_users, num_shards, num_imgs, ratio=1)
    elif args.ill_case == 2:
        return noniid_label_2(dataset, args.num_users, int(num_shards * 2), int(num_imgs / 2))
    elif args.ill_case == 3:
        return noniid_ratio_r_label_1(dataset, args.num_users, num_shards, num_imgs, ratio=0.8)
    elif args.ill_case == 4:
        return noniid_ratio_r_label_1(dataset, args.num_users, num_shards, num_imgs, ratio=0.5)
    else:
        print("Error Non-IID case")


def noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio=1):
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # assign main class
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: int((rand + ratio) * num_imgs)]), axis=0)

    if ratio < 1:
        rest_idxs = np.array([], dtype='int64')
        idx_shard = [i for i in range(num_shards)]
        for i in idx_shard:
            rest_idxs = np.concatenate((rest_idxs, idxs[int((i + ratio) * num_imgs): (i + 1) * num_imgs]), axis=0)

        num_items = int(len(dataset) / num_users * (1 - ratio))
        for i in range(num_users):
            rest_to_add = set(np.random.choice(rest_idxs, num_items, replace=False))
            dict_users[i] = np.concatenate((dict_users[i], list(rest_to_add)), axis=0)
            rest_idxs = list(set(rest_idxs) -rest_to_add)

    return dict_users


def noniid_label_2(dataset, num_users, num_shards, num_imgs):
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)

    return dict_users


def noniid_dirichlet(dataset, num_users, num_classes, alpha=0.4):
    y_train = np.array(dataset.targets)

    min_size_train = 0
    min_require_size = 10
    K = num_classes

    N_train = len(y_train)
    dict_users = {}

    while min_size_train < min_require_size:
        idx_batch_train = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k_train = np.where(y_train == k)[0]
            np.random.shuffle(idx_k_train)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions_train = np.array([p * (len(idx_j) < N_train / num_users) for p, idx_j in zip(proportions, idx_batch_train)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k_train)).astype(int)[:-1]
            idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, proportions_train))]
            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])

    for j in range(num_users):
        np.random.shuffle(idx_batch_train[j])
        dict_users[j] = idx_batch_train[j]

    return dict_users


def mnist_iid(dataset, args):
    return iid(dataset, args)


def mnist_noniid(dataset, args):
    num_shards, num_imgs = 100, 600
    return non_iid(dataset, args, num_shards, num_imgs)


def cifar10_iid(dataset, args):
    return iid(dataset, args)


def cifar10_noniid(dataset, args):
    num_shards, num_imgs = 100, 500
    return non_iid(dataset, args, num_shards, num_imgs)


def cinic10_noniid(dataset, args):
    num_shards, num_imgs = 100, 900
    return non_iid(dataset, args, num_shards, num_imgs)


def cifar100_iid(dataset, args):
    return iid(dataset, args)


def cifar100_noniid(dataset, args):
    num_shards, num_imgs = 100, 500
    return non_iid(dataset, args, num_shards, num_imgs)




