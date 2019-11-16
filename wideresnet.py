import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pruning.layers import MaskedLinear, MaskedConv2d, Flatten, MaskedBN1d, MaskedBN2d
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = MaskedBN2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MaskedBN2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and MaskedConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Flatten_nChannels(nn.Module):
    def __init__(self, nChannels):
        super(Flatten_nChannels, self).__init__()
        self.nChannels = nChannels

    def forward(self, x):
        return x.view(-1, self.nChannels)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = MaskedConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MaskedBN2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = MaskedLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.model = nn.Sequential(self.conv1, 
                            self.block1.layer,
                            self.block2.layer,
                            self.block3.layer,
                            self.bn1,
                            self.relu,
                            nn.AvgPool2d(8),
                            Flatten_nChannels(self.nChannels),
                            self.fc,
            )
        self.index = 0
        for m in self.model.modules():
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

    def forward(self, x):
        '''
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        '''
        out = self.model(x)
        return out


    def set_masks(self, masks, transfer=False):
        def set_layer_mask(layer, masks, index):
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                #print(layer.__class__.__name__)
                layer.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] +=1
            if isinstance(layer, nn.Sequential):
                for i in layer.children():
                    #print(i.__class__.__name__)
                    set_layer_mask(i, masks, index)
            if isinstance(layer, BasicBlock):
                for i in layer.children():
                    #print(i.__class__.__name__)
                    set_layer_mask(i, masks, index)


        index = np.array([0])
        
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for i in layer.children():
                    #print(i.__class__.__name__)
                    set_layer_mask(i, masks, index)
            else:
                set_layer_mask(layer, masks, index)
        assert index[0] == self.index, "seif.index["+str(self.index)+"] !="\
                                "mask index["+str(index[0])+"]"











