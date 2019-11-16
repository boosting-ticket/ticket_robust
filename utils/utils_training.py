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


def updateBN(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(0.0001*torch.sign(m.weight.data))

def train_one_epoch(m, masks, loss_fn, optimizer, loader_train, args, verbose=False):
    model = m.model
    
    model.train()
    for t, (x, y) in enumerate(loader_train):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        scores = model(x)
        loss = loss_fn(scores, y)
        optimizer.zero_grad()
        loss.backward()
        #updateBN(model)
        optimizer.step()
        m.set_masks(masks, transfer=args.transfer)# apply mask
        
        if args.create_init:
            if args.init_step == 0:
                torch.save(m.model.state_dict(), args.init_path)
                print("init", args.init_step, "model saved to", args.init_path)
                exit()
            else:
                args.init_step -= 1
        
        if (verbose > 0) and (t % verbose == 0):
            acc = float((scores.max(1)[1]==y).sum())/float(y.shape[0])
            print('Batch [%d/%d] training loss = %.4f, training acc = %.2f' % (t, len(loader_train), loss.data, acc))


def train_one_epoch_l1(m, masks, loss_fn, optimizer, loader_train, args, verbose=False):
    model = m.model
    
    model.train()
    for t, (x, y) in enumerate(loader_train):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        scores = model(x)
        loss = loss_fn(scores, y)
        l1_loss = 0.0
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1_loss = l1_loss + (1e-5 * torch.sum(torch.abs(param)))
        loss = loss + l1_loss
        optimizer.zero_grad()
        loss.backward()
        #updateBN(model)
        optimizer.step()
        m.set_masks(masks, transfer=args.transfer)# apply mask
        
        if args.create_init:
            if args.init_step == 0:
                torch.save(m.model.state_dict(), args.init_path)
                print("init", args.init_step, "model saved to", args.init_path)
                exit()
            else:
                args.init_step -= 1
        
        if (verbose > 0) and (t % verbose == 0):
            acc = float((scores.max(1)[1]==y).sum())/float(y.shape[0])
            print('Batch [%d/%d] training loss = %.4f, training acc = %.2f' % (t, len(loader_train), loss.data, acc))



def test(model, loader, loss_fn):
    model.eval()
    l_avg = 0
    num_samples = 0
    num_correct = 0

    for i, (x, y) in enumerate(loader):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        scores = model(x)
        preds = scores.max(1)[1]
        num_correct += (preds == y).sum().item()
        num_samples += int(x.shape[0])
        l_avg += loss_fn(scores, y).item()
        
    acc = float(num_correct) / num_samples
    l_avg = float(l_avg) / float(i)

    print('Test accuracy: {:.2f}% ({}/{}), Test loss:{:.4f}'.format(
        100.*acc,
        num_correct,
        num_samples,
        l_avg
        ))
    return acc, l_avg



