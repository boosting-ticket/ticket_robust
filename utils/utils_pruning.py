import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from wideresnet import BasicBlock

import numpy as np
import time 

from pruning.methods import filter_prune, prune_one_filter
from pruning.utils import to_var, train, test, prune_rate, arg_nonzero_min
from pruning.layers import Flatten

from collections import OrderedDict

from utils.utils_attacks import *

'''
def get_init_masks(model):
    """
    return initial masks for the conv and fc layers. 
    It is assumed that both types of layers use their Masked counterparts in the seuqential model.
    """
    def init_layer(i, masks):
        if isinstance(i, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
            masks.append(np.ones_like(i.weight.data.cpu().numpy()))
        return masks
    
    masks = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for i in layer.children():
                init_layer(i, masks)
        else:
            init_layer(layer, masks)
    return masks
'''
'''
def get_init_masks(model):
    """
    return initial masks for the conv and fc layers. 
    It is assumed that both types of layers use their Masked counterparts in the seuqential model.
    """
    def init_layer(layer, masks):
        #print(layer.__class__.__name__)
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
            masks.append(np.ones_like(layer.weight.data.cpu().numpy()))
        if isinstance(layer, nn.Sequential):
            for i in layer.children():
                #print(i.__class__.__name__)
                init_layer(i, masks)
        if isinstance(layer, BasicBlock):
            for i in layer.children():
                #print(i.__class__.__name__)
                init_layer(i, masks)
    
    masks = []
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for i in layer.children():
                init_layer(i, masks)
        else:
            init_layer(layer, masks)
    print("get_init_masks", len(masks))
    return masks
'''

def get_init_masks(model):
    """
    return initial masks for the conv and fc layers. 
    It is assumed that both types of layers use their Masked counterparts in the seuqential model.
    """
    masks = []
    for layer in model.modules():
        #print(m.__class__.__name__)
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            #print(layer.__class__.__name__)
            masks.append(np.ones_like(layer.weight.data.cpu().numpy()))
    return masks


            
def prune_one_filter_updated(model, masks, pruning_flags):
    '''
    masks (List):  Mask array for each layer. 
    pruning_flags: Flag for each convolutional to decide whethere to prune a filter.
    
    Ref: arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4:  # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

        if len(p.data.size()) == 2 and p.data.size()[0] != N_CLASSES:  # select fc layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1)/(p_np.shape[1])
            # normalization
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"
    assert len(masks) == len(pruning_flags), "something wrong here"

    values = np.array(values)

    for i in range(len(pruning_flags)):
        if pruning_flags[i]:
            to_prune_filter_ind = int(values[i, 1])
            masks[i][to_prune_filter_ind] = 0
    return masks


def create_pruning_flags(net, pruning_ratio, epochs):
    """
    pruning_ratio (array): pruning ratio for each layer 
    
    return: A array with number of filter needed to be pruned at each layer for every epoch. 
    """
    n_pruning_filter = []
    i = 0
    for p in net.parameters():
        if len(p.data.size()) == 4 or len(p.data.size()) == 2 and p.data.size()[0] != N_CLASSES:  # nasty way of selecting conv/fc layer
            p_np = p.data.cpu().numpy()
            n_pruning_filter.append(int(len(p_np)/100.0*pruning_ratio[i]))
            i += 1
    n_pruning_filter = np.array(n_pruning_filter)

    flags = np.zeros([len(n_pruning_filter), epochs])
    steps = np.ceil(n_pruning_filter/epochs)
    
    for i in range(len(n_pruning_filter)):
        flags[i].fill(steps[i])
        s = np.cumsum(flags[i]) - n_pruning_filter[i]
        index = np.argmax(s >= 0)
        flags[i][index:] = 0
    return flags


def filter_prune_updated(model, masks, pruning_steps_this_epoch):
    '''
    pruning_steps_this_epoch (List): Number of filters need to pruned at each layer for this epoch. 
    '''
    max_step = int(np.max(pruning_steps_this_epoch))

    pruning_flags_this_epoch = np.ones(
        [len(pruning_steps_this_epoch), max_step])
    for i in range(len(pruning_flags_this_epoch)):
        pruning_flags_this_epoch[i, int(pruning_steps_this_epoch[i]):] = 0

    for i in range(pruning_flags_this_epoch.shape[-1]):
        masks = prune_one_filter_updated(
            model, masks, pruning_flags_this_epoch[:, i])
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
    print('{:.2f}% pruned'.format(current_pruning_perc))

    return masks
