import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test(args, net, dataset_test):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = []
    test_accuracy = 0
    test_loader = DataLoader(dataset_test, batch_size=args.bs)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)['output']
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())
            accuracy = (outputs.argmax(1) == labels).sum()
            test_accuracy += accuracy.item()
    return sum(test_loss) / len(test_loss), test_accuracy / len(dataset_test)
