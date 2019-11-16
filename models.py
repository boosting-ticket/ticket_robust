import torch
import torch.nn as nn
from torch.autograd import Variable
from pruning.layers import MaskedLinear, MaskedConv2d, Flatten, MaskedBN1d, MaskedBN2d
import math
import numpy as np
import pdb

class cifar_model_large():
    def __init__(self, width, num_classes=10):
        self.num_classes = num_classes
        self.model = nn.Sequential(
            MaskedConv2d(3, 4*width, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(4*width, 4*width, 4, stride=2, padding=1),
            nn.ReLU(),
            MaskedConv2d(4*width, 8*width, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(8*width, 8*width, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            MaskedLinear(8*width*8*8, 512),
            nn.ReLU(),
            MaskedLinear(512, 512),
            nn.ReLU(),
            MaskedLinear(512, self.num_classes)
        )
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        index = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.set_mask(torch.from_numpy(masks[index]))
                index += 1
        assert index == len(masks) #Again a hack to make sure that masks are provided for each layer.



class cifar_model_small():
    def __init__(self, width, num_classes=10):
        self.num_classes = num_classes
        self.model = nn.Sequential(
            MaskedConv2d(3, 4*width, 3, stride=2, padding=1),
            nn.ReLU(),
            MaskedConv2d(4*width, 8*width, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            MaskedLinear(8*width*8*8, 512),
            nn.ReLU(),
            MaskedLinear(512, 512),
            nn.ReLU(),
            MaskedLinear(512, self.num_classes)
        )
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        index = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.set_mask(torch.from_numpy(masks[index]))
                index += 1
        assert index == len(masks) #Again a hack to make sure that masks are provided for each layer.


class cifar_conv2():
    def __init__(self, depth=1, num_classes=10):
        self.num_classes = num_classes
        self.model = nn.Sequential(
            MaskedConv2d(3, 64*depth, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(64*depth, 64*depth, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten(),
            MaskedLinear(64*depth*16*16, 256),
            nn.ReLU(),
            MaskedLinear(256, 256),
            nn.ReLU(),
            MaskedLinear(256, 256),
            nn.ReLU(),
            MaskedLinear(256, self.num_classes)
        )
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        index = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.set_mask(torch.from_numpy(masks[index]))
                index += 1
        assert index == len(masks) #Again a hack to make sure that masks are provided for each layer.


class cifar_conv4():
    def __init__(self, depth=1, num_classes=10):
        self.num_classes = num_classes
        self.model = nn.Sequential(
            MaskedConv2d(3, 64*depth, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(64*depth, 64*depth, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            MaskedConv2d(64*depth, 128*depth, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(128*depth, 128*depth, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten(),
            MaskedLinear(128*depth*8*8, 256),
            nn.ReLU(),
            MaskedLinear(256, 256),
            nn.ReLU(),
            MaskedLinear(256, 256),
            nn.ReLU(),
            MaskedLinear(256, self.num_classes)
        )
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        index = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.set_mask(torch.from_numpy(masks[index]))
                index += 1
        assert index == len(masks) #Again a hack to make sure that masks are provided for each layer.



class cifar_conv4_bn_max2():
    def __init__(self, depth=1, num_classes=10):
        self.num_classes = num_classes
        self.model = nn.Sequential(
            MaskedConv2d(3, 64*depth, 3, stride=1, padding=1),
            MaskedBN2d(64*depth),
            nn.ReLU(),
            MaskedConv2d(64*depth, 64*depth, 3, stride=1, padding=1),
            MaskedBN2d(64*depth),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            MaskedConv2d(64*depth, 128*depth, 3, stride=1, padding=1),
            MaskedBN2d(128*depth),
            nn.ReLU(),
            MaskedConv2d(128*depth, 128*depth, 3, stride=1, padding=1),
            MaskedBN2d(128*depth),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten(),
            MaskedLinear(128*depth*8*8, 256),
            MaskedBN1d(math.ceil(256)),
            nn.ReLU(),
            MaskedLinear(256, 256),
            MaskedBN1d(math.ceil(256)),
            nn.ReLU(),
            MaskedLinear(256, 256),
            MaskedBN1d(math.ceil(256)),
            nn.ReLU(),
            MaskedLinear(256, self.num_classes)
        )
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def set_masks(self, masks):
        def set_layer_mask(i, masks, index):
            if isinstance(i, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                i.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] +=1 

        index = np.array([0])
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for i in layer.children():
                    set_layer_mask(i, masks, index)
            else:
                set_layer_mask(layer, masks, index)


class all_cnn():
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = nn.Sequential(
            MaskedConv2d(3, 96, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(96, 96, 3, stride=2, padding=1),
            nn.ReLU(),
            MaskedConv2d(96, 192, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(192, 192, 3, stride=2, padding=1),
            nn.ReLU(),
            MaskedConv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(192, 192, 1, stride=1),
            nn.ReLU(),
            MaskedConv2d(192, 32, 1, stride=1),
            nn.ReLU(),
            Flatten(),
            MaskedLinear(8*8*32, self.num_classes)
        )
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                
    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        index = 0
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.set_mask(torch.from_numpy(masks[index]))
                index += 1
        assert index == len(masks) #Again a hack to make sure that masks are provided for each layer.
       
    
def cifar_model_large_wong(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


defaultcfg = {
    7 : [64, 'M', 128, 'M', 256, 'M', 512],
    '8s' : [32, 'M', 32, 'M', 64, 'M', 64, 'M', 64],
    '8m' : [32, 'M', 32, 'M', 64, 'M', 128, 'M', 128],
    8 : [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512],
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg(nn.Module):
    # ref: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/l1-norm-pruning/models/vgg.py
    def __init__(self, dataset='cifar', depth=16, init_weights=True, cfg=None, cap_ratio=1.0):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg
        self.depth = depth
        if depth == "8s": self.depth = 8
        self.cap_ratio = cap_ratio
        self.index = 0
        
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
              MaskedLinear(math.ceil(cfg[-1]*cap_ratio), math.ceil(512*cap_ratio)),
              MaskedBN1d(math.ceil(512*cap_ratio)),
              nn.ReLU(inplace=True),
              MaskedLinear(math.ceil(512*cap_ratio), num_classes)
            )
        self.model = nn.Sequential(
            self.feature, nn.AvgPool2d(2), Flatten(), self.classifier)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = math.ceil(v*self.cap_ratio)
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, MaskedBN2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        '''
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.model(x)
        '''
        return y

    def _initialize_weights(self):
        for m in self.model.modules():
            #print(self.index, m.__class__.__name__)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #print(n, m.weight.data.normal_(0, math.sqrt(2. / n))[0])
                if m.bias is not None:
                    m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                self.index += 1
            elif isinstance(m, nn.BatchNorm1d):
                self.index += 1
                
    def set_masks(self, masks, transfer=False):
        def set_layer_mask(i, masks, index):
            if isinstance(i, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                #print(index[0], i.__class__.__name__)
                if transfer and index[0]>=(masks.shape[0]-1):
                    #print(i.weight.shape, masks[index[0]].shape)
                    pass
                else:
                    #print(masks[index[0]].shape)
                    i.set_mask(torch.from_numpy(masks[index[0]]))
                    #print(i.weight.shape)
                index[0] += 1

        index = np.array([0])
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for i in layer.children():
                    set_layer_mask(i, masks, index)
            else:
                set_layer_mask(layer, masks, index)
        assert index[0] == 2*self.depth - 3 ==self.index #Again a hack to make sure that masks are provided for each layer.


    def transfer_model(self, net_orig):
        index = 0
        orig_index = 0
        orig_layers = []

        for layer in net_orig.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                #print(layer.__class__.__name__)
                orig_layers.append(layer)
                orig_index+=1
        
        for layer in self.model.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                #print(layer.__class__.__name__)

                if index >= len(orig_layers)-1:
                    #print("last", index, layer.weight.data.shape, orig_layers[index].weight.data.shape)
                    layer.weight.data = orig_layers[index].weight.data.clone().repeat(10, 1).detach()

                if index < len(orig_layers)-1: 
                    layer.weight.data = orig_layers[index].weight.data.clone().detach()
                    #print(index, layer.weight.data.shape, orig_layers[index].weight.data.shape)
                    index+=1
                    
       





class vgg_no_bn(nn.Module):
    # ref: https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/l1-norm-pruning/models/vgg.py
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None, cap_ratio=1.0):
        super(vgg_no_bn, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg
        self.depth = depth
        if depth == "8s": self.depth = 8
        self.cap_ratio = cap_ratio
        
        self.feature = self.make_layers(cfg, False)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
              MaskedLinear(math.ceil(cfg[-1]*cap_ratio), math.ceil(512*cap_ratio)),
              #MaskedBN1d(math.ceil(512*cap_ratio)),
              nn.ReLU(inplace=True),
              MaskedLinear(math.ceil(512*cap_ratio), num_classes)
            )
        self.model = nn.Sequential(
            self.feature, nn.AvgPool2d(2), Flatten(), self.classifier)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = math.ceil(v*self.cap_ratio)
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, MaskedBN2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        '''
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.model(x)
        '''
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def set_masks(self, masks):
        def set_layer_mask(i, masks, index):
            if isinstance(i, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
                i.set_mask(torch.from_numpy(masks[index[0]]))
                index[0] +=1 

        index = np.array([0])
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for i in layer.children():
                    set_layer_mask(i, masks, index)
            else:
                set_layer_mask(layer, masks, index)
        #assert index[0] == 2*self.depth - 3 #Again a hack to make sure that masks are provided for each layer.

if __name__ == '__main__':
    import numpy as np
    import random
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic=True
    net = vgg(depth=16).cuda()
    x = Variable(torch.ones(16, 3, 40, 40)).cuda()
    y = net(x)
    print(y)
