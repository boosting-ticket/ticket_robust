'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
from pruning.layers import MaskedLinear, MaskedConv2d, Flatten, MaskedBN1d, MaskedBN2d
import math
import numpy as np
import pdb
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MaskedBN2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MaskedBN2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                MaskedBN2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MaskedBN2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MaskedBN2d(planes)
        self.conv3 = MaskedConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = MaskedBN2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                MaskedBN2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = MaskedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MaskedBN2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = MaskedLinear(512*block.expansion, num_classes)
        
        self.index = 0
        for m in self.modules():
            #print(m.__class__.__name__)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                self.index += 1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                self.index += 1
            else:
                pass
        #print("index init:", self.index)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.model(x)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = ResNet(BasicBlock, [2,2,2,2])
        self.index = self.model.index

    def forward(self, x):
        out = self.model(x)
        return out

    def set_masks(self, masks, transfer=False):
        
        index = np.array([0])

        for layer in self.model.modules():
            #print(m.__class__.__name__)
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                #print(layer.__class__.__name__)
                layer.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] +=1

        assert index[0] == self.index, "seif.index["+str(self.index)+"] !="\
                                "mask index["+str(index[0])+"]"



class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = ResNet(BasicBlock, [3,4,6,3])
        self.index = self.model.index

    def forward(self, x):
        out = self.model(x)
        return out

    def set_masks(self, masks, transfer=False):
        
        index = np.array([0])

        for layer in self.model.modules():
            #print(m.__class__.__name__)
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                #print(layer.__class__.__name__)
                layer.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] +=1

        assert index[0] == self.index, "seif.index["+str(self.index)+"] !="\
                                "mask index["+str(index[0])+"]"




def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
