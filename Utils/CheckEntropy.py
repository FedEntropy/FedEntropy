from math import log

import numpy as np
import torch
import copy


def get_entropy(l):
    entropy = 0.0
    for i in range(len(l)):
        entropy += - l[i] * log(l[i], 2)
    return entropy


def check_entropy_sub(args, w_locals, lens, soft_label_list, idxs):
    n = max(int(args.num_users * args.frac), 1)
    soft_label_list = torch.reshape(soft_label_list, (n, args.num_classes))
    new_a = torch.clone(soft_label_list)
    delList = []

    # while n > 0.5 * args.num_users * args.frac:
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
            delList.append(idxs[i])

    # lens = []
    # for i in range(len(w_locals)):
    #     lens.append(get_entropy(soft_label_list[i]))
    #
    # t = sum(lens)
    # for i in range(len(lens)):
    #     lens[i] = lens[i] / t

    return w_locals, lens, np.array(delList)


def check_entropy_add(args, w_locals, lens, soft_label_list, idxs):
    n = 0
    soft_label_list = torch.reshape(soft_label_list, (n, args.num_classes))
    #
    # for i in range(len(w_locals)):
    #     lens.append(get_entropy(soft_label_list[i]))
    #
    # for

    return w_locals, lens
