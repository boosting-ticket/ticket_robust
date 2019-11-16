import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.utils import to_var
import math


class MaskedBN2d(nn.BatchNorm2d):
    def __init__(self, out_channels):
        super(MaskedBN2d, self).__init__(out_channels)
        self.out_channels = out_channels
        self.mask_flag = False
        
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.bias.data = self.bias.data*self.mask.data
        self.mask_flag = True
        
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training)
        
      
    
class MaskedBN1d(nn.BatchNorm1d):
    def __init__(self, out_channels):
        super(MaskedBN1d, self).__init__(out_channels)
        self.out_channels = out_channels
        self.mask_flag = False
        
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.bias.data = self.bias.data*self.mask.data
        self.mask_flag = True
        
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training)
        


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.bias.data = self.bias.data*self.mask.data.mean(1)
        self.mask_flag = True
    
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        self.bias_flag = bias
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels 
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if self.bias_flag:
            self.bias.data.zero_()
        
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        if self.bias_flag:
                self.bias.data = self.bias.data*self.mask.data.sum((1, 2, 3))/(self.mask.shape[1]*self.mask.shape[2]*self.mask.shape[3])
        self.mask_flag = True
    
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
