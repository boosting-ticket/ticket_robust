import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np
import time 
import os

from pruning.methods import filter_prune, prune_one_filter
from pruning.utils import to_var, train, test, prune_rate, arg_nonzero_min
from pruning.layers import Flatten

from collections import OrderedDict

from utils.utils_attacks import * 
from utils.utils_pruning import *
from utils.utils_mixtrain import *


def train_one_noise_epoch(m, masks, loss_fn, noise_sd, optimizer, loader_train, verbose=False):
    if isinstance(m, nn.Module):
        # if the model class is inherited from nn.modules.
        model = m
    else:
        model = m.model
    '''
    torch.save(m.model.state_dict(), "pure_vgg16_init100.pth")
    print("original init model saved to", "pure_vgg16_init100.pth")
    '''
    model.train()
    for t, (x, y) in enumerate(loader_train):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        x = x + torch.randn_like(x, device='cuda') * noise_sd

        #x_var, y_var = to_var(x), to_var(y.long())
        scores = model(x)
        loss = loss_fn(scores, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        m.set_masks(masks)# apply mask
        '''
        if t == 100:
            torch.save(m.model.state_dict(), "nat_vgg16_init100.pth")
            print("init 100 model saved to", "nat_vgg16_init100.pth")
        '''
        if verbose and (t % 100 == 0):
            acc = float((scores.max(1)[1]==y).sum())/float(y.shape[0])
            print('Batch [%d/%d] training loss = %.8f, training acc = %.8f' % (t, len(loader_train), loss.data, acc))


def test_noise(model, noise_sd, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    for x, y in loader:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        x = x + torch.randn_like(x, device='cuda') * noise_sd
        #x_var = to_var(x)
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / num_samples

    print('Test accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
        ))
    return acc



