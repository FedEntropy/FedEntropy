import numpy as np
from models.Update import LocalUpdate_Fast
from models.test import test_img
from models.Fed import *
from math import log

import random


def DynamicEntropy(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()
    acc = []

    a = np.arange(args.num_users)
    positive = set(a)
    negative = set()

    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter))

        k = random.random()
        m = max(int(args.num_users * args.frac), 1)
        w_locals = []
        soft_label_locals = None
        lens = []

        branch = True
        if k > 0.8 and len(negative) > m - 1 or len(positive) < m:
            branch = False
            print("negative")
            choice = random.sample(negative, m)
        else:
            branch = True
            print("positive")
            choice = random.sample(positive, m)

        idxs_users = np.array(choice)
        #w_glob = net_glob.state_dict()
        for idx in idxs_users:
            local = LocalUpdate_Fast(args, dataset_train, dict_users[idx])
            w, soft_label = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            if soft_label_locals is None:
                soft_label_locals = soft_label
            else:
                soft_label_locals = torch.cat((soft_label_locals, soft_label), 0)

        w_locals, lens, delList = check_entropy(args, w_locals, lens, soft_label_locals)
        if branch:
            # positive
            negative = negative.union(idxs_users[delList])
            positive.difference_update(idxs_users[delList])
        else:
            c = []
            for j in range(len(idxs_users)):
                if j not in delList:
                    c.append(j)
            positive = positive.union(idxs_users[c])
            negative.difference_update(idxs_users[c])

        print("positive: ", positive)
        print("negative: ", negative)

        w_glob = Aggregation(w_locals, lens)
        net_glob.load_state_dict(w_glob)
        accuracy, loss = test_img(net_glob, dataset_test, args)
        print(accuracy)


def get_entropy(l):
    entropy = 0.0
    for i in range(len(l)):
        entropy += - l[i] * log(l[i], 2)
    return entropy


def check_entropy(args, w_locals, lens, soft_label_list):
    n = max(int(args.num_users * args.frac), 1)
    soft_label_list = torch.reshape(soft_label_list, (n, args.num_classes))
    new_a = torch.clone(soft_label_list)
    delList = []

    while n > 0:
        c = torch.clone(soft_label_list)
        for i in range(len(lens)):
            c[i] *= lens[i]
        maxE = get_entropy(torch.sum(c, dim=0) / sum(lens))

        index = -1
        for i in range(n):
            new_lens = copy.deepcopy(lens)
            del new_lens[i]
            l = [x for x in range(n)]
            l.remove(i)

            c = torch.clone(soft_label_list[l])
            for j in range(len(new_lens)):
                c[j] *= new_lens[j]
            entropy = get_entropy(torch.sum(c, dim=0) / sum(new_lens))

            if maxE < entropy:
                maxE = entropy
                index = i

        if index == -1:
            break
        else:
            l = [x for x in range(n)]
            l.remove(index)
            soft_label_list = soft_label_list[l]
            n -= 1
            del w_locals[index]
            del lens[index]

    for i in range(len(new_a)):
        if new_a[i] not in soft_label_list:
            delList.append(i)
    return w_locals, lens, delList
