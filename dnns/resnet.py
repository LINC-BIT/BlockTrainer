from torchvision.models import resnet50 as _resnet50, resnet18 as _resnet18
from torch import nn 


def resnet50(num_classes, pretrained=True):
    res = _resnet50(pretrained=pretrained)
    res.fc = nn.Linear(2048, num_classes, bias=True)
    return res


def resnet18(num_classes, pretrained=True):
    res = _resnet18(pretrained=pretrained)
    res.fc = nn.Linear(512, num_classes, bias=True)
    return res
