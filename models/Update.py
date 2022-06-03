import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Fed import *
from torch.optim import Optimizer


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate_FedAvg(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.dataset = dataset

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()


class LocalUpdate_FedProx(object):
    def __init__(self, args, glob_model, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                predict_loss = self.loss_func(log_probs, labels)

                penalize_loss = 0.0
                for param_index, param in enumerate(net.parameters()):
                    penalize_loss += (
                            (self.prox_alpha / 2) * torch.norm(param - global_weight_collector[param_index]) ** 2)

                loss = predict_loss + penalize_loss
                loss.backward()
                optimizer.step()

        return net.state_dict()


class LocalUpdate_Scaffold(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net, global_control, local_controls):
        local_net = copy.deepcopy(net)
        local_control = copy.deepcopy(local_controls)
        local_net.train()
        param_groups = []
        for para in local_net.parameters():
            param_groups.append({'params': para})
        optimizer = SCAFFOLDOptimizer(param_groups, lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                local_net.zero_grad()
                log_probs = local_net(images)
                loss = self.loss_func(log_probs, labels)

                loss.backward()
                optimizer.step(global_control, local_control)

        delta_weight = mysub(local_net.state_dict(), net.state_dict())
        # penalty = mymul(delta_weight, 0.05)
        penalty = mydiv(delta_weight, int(1 / 0.05))
        new_control = mysub(mysub(local_control.state_dict(), global_control.state_dict()), penalty)

        return local_net.state_dict(), new_control


class LocalUpdate_History(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()


class LocalUpdate_Fast(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            soft_label = None
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_prob = torch.softmax(net(images), dim=1).detach().cpu()
                if soft_label is None:
                    soft_label = log_prob
                else:
                    soft_label = torch.cat((soft_label, log_prob), 0)

            soft_label = torch.sum(soft_label, dim=0) / soft_label.shape[0]
        return net.state_dict(), soft_label


class LocalUpdate_Fast_Prox(object):
    def __init__(self, args, glob_model, dataset, idxs):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True)
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                predict_loss = self.loss_func(log_probs, labels)

                penalize_loss = 0.0
                for param_index, param in enumerate(net.parameters()):
                    penalize_loss += (
                            (self.prox_alpha / 2) * torch.norm(param - global_weight_collector[param_index]) ** 2)

                loss = predict_loss + penalize_loss
                loss.backward()
                optimizer.step()

        net.eval()
        soft_label = None
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_prob = torch.softmax(net(images), dim=1)
            if soft_label is None:
                soft_label = log_prob
            else:
                soft_label = torch.cat((soft_label, log_prob), 0)

        soft_label = torch.sum(soft_label, dim=0) / soft_label.shape[0]
        return net.state_dict(), soft_label


class LocalUpdate_Moon(object):
    def __init__(self, args, glob_model, old_models, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.glob_model = glob_model
        self.old_models = old_models
        self.contrastive_alpha = args.contrastive_alpha
        self.temperature = args.temperature

    def train(self, net):

        self.glob_model.eval()
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        Predict_loss = 0
        Contrastive_loss = 0

        for iter in range(self.args.local_ep):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                output = net(images)
                predictive_loss = self.loss_func(output['output'], labels)

                output_representation = output['representation']
                pos_representation = self.glob_model(images)['representation']
                posi = self.cos(output_representation, pos_representation)
                logits = posi.reshape(-1, 1)

                for previous_net in self.old_models:
                    neg_representation = previous_net(images)['representation']
                    nega = self.cos(output_representation, neg_representation)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= self.temperature
                labels = torch.zeros(images.size(0)).to(self.args.device).long()

                contrastive_loss = self.contrastive_alpha * self.loss_func(logits, labels)

                loss = predictive_loss + contrastive_loss
                Predict_loss += predictive_loss.item()
                Contrastive_loss += contrastive_loss.item()

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(predictive_loss.item())
                epoch_loss2_collector.append(contrastive_loss.item())

            # epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            # epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            # epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

        net.eval()
        soft_label = None
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_prob = torch.softmax(net(images)['output'], dim=1)
            if soft_label is None:
                soft_label = log_prob
            else:
                soft_label = torch.cat((soft_label, log_prob), 0)

        soft_label = torch.sum(soft_label, dim=0) / soft_label.shape[0]
        return net.state_dict(), soft_label

        # return net.state_dict()


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, momentum):
        default = dict(lr=lr, momentum=momentum)
        super(SCAFFOLDOptimizer, self).__init__(params, default)
        pass

    def step(self, global_control, local_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group, c, ci in zip(self.param_groups, global_control.parameters(), local_control.parameters()):
            p = group['params'][0]
            momentum = group['momentum']
            if p.grad is None:
                continue
            d_p = p.grad.data + c.data - ci.data

            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1)
                d_p = buf
            p.data.add_(d_p, alpha=-group['lr'])

        return loss


def mydiv(w, a):
    out = copy.deepcopy(w)
    for k in out.keys():
        # for i in range(1, a):
        #     out[k] += w[k]
        out[k] = torch.div(out[k], a)
    return out

def mysub(w1, w2):
    out = copy.deepcopy(w1)
    for k in out.keys():
        # for i in range(1, len(w)):
        out[k] -= w2[k]
        # out[k] = torch.div(out[k], len(w))
    return out

def mymul(w, a):
    out = copy.deepcopy(w)
    for k in out.keys():
        # for i in range(1, a):
        #     out[k] += w[k]
        out[k] = torch.mul(out[k], a)
    return out
