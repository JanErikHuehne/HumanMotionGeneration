import torch

import torch.nn as nn
import torchvision.models as models
torch.autograd.set_detect_anomaly(True)
class ResNetCNN(nn.Module):
    def __init__(self, ttype="joints"):
        super(ResNetCNN, self).__init__()
        if ttype == "joints":
            self.resnet = models.resnet18(pretrained=False, num_classes=23*3)
        elif ttype == "poses":
            self.resnet = models.resnet18(pretrained=False, num_classes=24*3)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
      

    def forward(self, x):
        x = self.resnet(x)
        return x

class VGGCNN(nn.Module):
    def __init__(self, ttype="joints"):
        super(VGGCNN, self).__init__()
        if ttype == "joints":
            self.vgg = models.vgg11(pretrained=False, num_classes=23*3)
        elif ttype == "poses":
            self.vgg = models.vgg11(pretrained=False, num_classes=24*3)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.vgg(x)
