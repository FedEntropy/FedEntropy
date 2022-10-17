from Model.LeNet import LeNet
from Model.MobileNetV2 import MobileNetV2
from Model.VGG import VGG
from Model.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152


def get_model(args):
    if args.model == "LeNet":
        net = LeNet(args).to(args.device)
        return net
    if args.model == "MobileNet":
        net = MobileNetV2(args).to(args.device)
        return net
    if args.model == "VGG":
        net = VGG(args).to(args.device)
        return net
    if args.model == "ResNet":
        net = resnet18(args).to(args.device)
        return net
