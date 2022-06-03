import copy
from math import log

import torch


def Aggregation(w, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg


def AggregationForAsyn(w, N):
    w_avg = None

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], N)

    return w_avg


def FedDelta(w1, w2):
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] -= w2[key]
    return w


def FedAdd(w1, w2):
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] += w2[key]
    return w


def FedHistory(w, historyModel):
    w_r = copy.deepcopy(w)
    for key in w_r.keys():
        delta = 0.25
        w_r[key] = w_r[key] + historyModel[-1][key] * 0.5
        for i in range(len(historyModel) - 1, 0):
            w_r[key] += delta * historyModel[i][key]
            delta *= 0.5
    return w_r


def FedMix(w, historyModel):
    w_r = copy.deepcopy(w)
    for key in w_r.keys():
        w_r[key] = w_r[key] * 0.8 + historyModel[key] * 0.2
    return w_r


def FedMixHistory(args, w, soft_label, lens, historyModel, historyLabel, historyLen):
    # pick 1 of 3
    w_r = copy.deepcopy(w)
    maxE = get_entropy(soft_label)
    index = -1

    for i in range(len(historyLabel)):
        l = torch.clone(soft_label)
        for j in range(args.num_classes):
            l[j] = l[j] * lens + historyLabel[i][j] * historyLen[i]
        entropy = get_entropy(l / (lens + historyLen[i]))
        if entropy > maxE:
            maxE = entropy
            index = i

    print(index)
    if index != -1:
        for key in w_r.keys():
            w_r[key] = w_r[key] * 0.9 + historyModel[index][key] * 0.1
    return w_r


def get_entropy(l):
    entropy = 0.0
    for i in range(len(l)):
        entropy += - l[i] * log(l[i], 2)
    return entropy


def FedHistory1(historyModel):
    w_r = copy.deepcopy(historyModel[-1])
    for key in w_r.keys():
        delta = 0.2
        w_r[key] = w_r[key] * 0.8
        for i in range(len(historyModel) - 1, 0):
            w_r[key] += delta * historyModel[i][key]
            delta *= 0.5
    return w_r


def Fed_InnerProduct(w1, w2):
    InnerProduct = 0.0
    for key in w1.keys():
        InnerProduct += w1[key] * w2[key]
    return InnerProduct
