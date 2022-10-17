import copy
import torch
import numpy as np


def Avg(w_locals, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w_locals)):
        if i == 0:
            w_avg = copy.deepcopy(w_locals[0])
            for k in w_avg.keys():
                w_avg[k] = w_locals[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w_locals[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg


def Avg_c(c_locals, lens, delta_c_sum):
    for i in range(len(c_locals)):
        delta_c_sum += c_locals[i] * lens[i]
    delta_c_sum = delta_c_sum / sum(lens)
    return delta_c_sum
