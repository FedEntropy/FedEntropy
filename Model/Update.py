import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):

    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return image, label


class Local_FedAvg(object):

    def __init__(self, args, dataset, idx):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(DatasetSplit(dataset, idx), batch_size=args.local_bs, shuffle=True,
                                       drop_last=False)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=self.args.lr)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = net(images)['output']
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return net.state_dict()


class Local_FedProx(object):

    def __init__(self, args, dataset, idx, net_glob):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(DatasetSplit(dataset, idx), batch_size=args.local_bs, shuffle=True,
                                       drop_last=False)
        self.net_glob = net_glob

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=self.args.lr)
        global_weight_collector = list(self.net_glob.parameters())

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = net(images)['output']
                predict_loss = self.criterion(outputs, labels)
                optimizer.zero_grad()

                penalize_loss = 0.0
                for param_index, param in enumerate(net.parameters()):
                    penalize_loss += (self.args.prox_alpha / 2) * torch.norm(
                        param - global_weight_collector[param_index]) ** 2

                loss = predict_loss + penalize_loss
                loss.backward()
                optimizer.step()

        return net.state_dict()


class LocalUpdate_Moon(object):
    def __init__(self, args, glob_model, old_models, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.glob_model = glob_model
        self.old_models = old_models
        self.contrastive_alpha = args.contrastive_alpha
        self.temperature = args.temperature
        self.verbose = verbose

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
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

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            if self.verbose:
                print('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (iter, epoch_loss, epoch_loss1, epoch_loss2))

        return net.state_dict()


class Local_FedEntropy(object):

    def __init__(self, args, dataset, idx):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(DatasetSplit(dataset, idx), batch_size=args.local_bs, shuffle=True,
                                       drop_last=False)

    def train(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=self.args.lr)

        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = net(images)['output']
                loss = self.criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        net.eval()
        soft_label = None
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_prob = torch.softmax(net(images)['output'], dim=1)
                if soft_label is None:
                    soft_label = log_prob
                else:
                    soft_label = torch.cat((soft_label, log_prob), 0)

        soft_label = torch.sum(soft_label, dim=0) / soft_label.shape[0]

        return net.state_dict(), soft_label
