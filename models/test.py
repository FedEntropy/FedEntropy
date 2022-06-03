import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, dataset, args):
    net_g.to(args.device)
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.bs)
    l = len(data_loader)

    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {: 4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))

    # net_g.to('cpu')
    # return accuracy, test_loss
    return accuracy, test_loss
