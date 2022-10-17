from Utils.SaveResult import save_result
from Model.Update import *
from Model.Test import test
from Model.Fed import *
from Utils.CheckEntropy import *
import numpy as np
import torch
import copy
import time
import random


def FedAvg(args, net_glob, train_dataset, test_dataset, dict_users):
    accuracy = []

    for round in range(args.epochs):
        print("####################{}####################".format(round))

        local_model = []
        lens = []

        start_time = time.time()
        m = max(int(args.frac * args.num_users), 1)
        idxs = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs)

        for idx in idxs:
            client = Local_FedAvg(args=args, dataset=train_dataset, idx=dict_users[idx])
            w = client.train(net=copy.deepcopy(net_glob).to(args.device))
            local_model.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

        w_avg = Avg(local_model, lens)
        net_glob.load_state_dict(w_avg)
        test_loss, test_accuracy = test(args, net_glob, test_dataset)

        accuracy.append(test_accuracy)

        finish_time = time.time()
        oneRoundTime = (finish_time - start_time)

        print('test_loss: {}, accuracy: {}, time: {}'.format(test_loss, test_accuracy, oneRoundTime))

    save_result(args, accuracy)


def FedProx(args, net_glob, train_dataset, test_dataset, dict_users):
    accuracy = []

    for round in range(args.epochs):
        print("####################{}####################".format(round))

        local_model = []
        lens = []

        start_time = time.time()
        m = max(int(args.frac * args.num_users), 1)
        idxs = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs)

        for idx in idxs:
            client = Local_FedProx(args=args, dataset=train_dataset, idx=dict_users[idx], net_glob=net_glob)
            w = client.train(net=copy.deepcopy(net_glob).to(args.device))
            local_model.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))

        w_avg = Avg(local_model, lens)
        net_glob.load_state_dict(w_avg)
        test_loss, test_accuracy = test(args, net_glob, test_dataset)

        accuracy.append(test_accuracy)
        finish_time = time.time()
        oneRoundTime = (finish_time - start_time)
        print('test_loss: {}, accuracy: {}, time: {}'.format(test_loss, test_accuracy, oneRoundTime))

    save_result(args, accuracy)


def moon(args, net_glob, train_dataset, test_dataset, dict_users):
    accuracy = []

    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
    old_nets_pool = [[] for i in range(args.num_users)]
    lens = [len(datasets) for _, datasets in dict_users.items()]

    for round in range(args.epochs):
        print("####################{}####################".format(round))

        start_time = time.time()
        w_glob = {}
        total_len = 0
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_Moon(args=args, glob_model=net_glob, old_models=old_nets_pool[idx],
                                     dataset=train_dataset, idxs=dict_users[idx])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            net_local.load_state_dict(w_local)

            w_local = local.train(net=net_local.to(args.device))

            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_local[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]

            total_len += len(dict_users[idx])

            if len(old_nets_pool[idx]) < args.model_buffer_size:
                old_net = copy.deepcopy(net_local)
                old_net.eval()
                for param in old_net.parameters():
                    param.requires_grad = False
                old_nets_pool[idx].append(old_net)
            else:
                old_net = copy.deepcopy(net_local)
                old_net.eval()
                for param in old_net.parameters():
                    param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[idx][i] = old_nets_pool[idx][i + 1]
                old_nets_pool[idx][args.model_buffer_size - 1] = old_net

        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        net_glob.load_state_dict(w_glob)
        test_loss, test_accuracy = test(args, net_glob, test_dataset)
        accuracy.append(test_accuracy)
        finish_time = time.time()
        oneRoundTime = (finish_time - start_time)

        print('test_loss: {}, accuracy: {}, time: {}'.format(test_loss, test_accuracy, oneRoundTime))

    save_result(args, accuracy)


def FedEntropy(args, net_glob, train_dataset, test_dataset, dict_users):
    accuracy = []

    value = []
    for i in range(args.num_users):
        value.append(1)

    for round in range(args.epochs):
        print("####################{}####################".format(round))

        p = []
        for i in range(len(value)):
            p.append(value[i] / sum(value))

        local_model = []
        lens = []
        soft_label_locals = None

        start_time = time.time()
        m = max(int(args.frac * args.num_users), 1)

        k = random.random()
        if k > args.threshold:
            idxs = np.random.choice(range(args.num_users), m, replace=False)
        else:
            idxs = np.random.choice(range(args.num_users), m, replace=False, p=p)

        # idxs = np.random.choice(range(args.num_users), m, replace=False, p=p)
        print(idxs)

        for idx in idxs:
            client = Local_FedEntropy(args=args, dataset=train_dataset, idx=dict_users[idx])
            w, soft_label = client.train(net=copy.deepcopy(net_glob).to(args.device))
            local_model.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            if soft_label_locals is None:
                soft_label_locals = soft_label
            else:
                soft_label_locals = torch.cat((soft_label_locals, soft_label), 0)

        local_model, lens, delList = check_entropy_sub(args, local_model, lens, soft_label_locals, idxs)
        agglist = []
        for idx in idxs:
            if idx not in delList:
                agglist.append(idx)
        print('agglist: ', agglist)
        print('dellist: ', delList)

        for idx in agglist:
            value[idx] += 1
        for idx in delList:
            value[idx] /= 2

        w_avg = Avg(local_model, lens)
        net_glob.load_state_dict(w_avg)
        test_loss, test_accuracy = test(args, net_glob, test_dataset)

        accuracy.append(test_accuracy)

        finish_time = time.time()
        oneRoundTime = (finish_time - start_time)

        print('test_loss: {}, accuracy: {}, time: {}'.format(test_loss, test_accuracy,
                                                                           oneRoundTime))

    save_result(args, accuracy)
