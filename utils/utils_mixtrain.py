import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np
import time

import argparse
from symbolic_interval.symbolic_network import sym_interval_analyze
from convex_adversarial import robust_loss, robust_loss_parallel
import math


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mixtrain_parallel_one_epoch(loader, m, masks, opt, epsilon, interval_weight,
                    verbose, clip_grad=None, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model = m.model
    model.train()
    
    end = time.time()
    use_cuda = torch.cuda.is_available()
    
    # Size of sampling
    #k = 1
    interval_weight = 50
    k = 10
    alpha = 0.75
    
    print("epsilon, alpha, interval_weight, lr, k:",epsilon, alpha,\
            interval_weight, opt.param_groups[0]['lr'], k)


    for i, (X,y) in enumerate(loader):
        robust_coin = False
        if(use_cuda):
            X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        data_time.update(time.time() - end)

        coin = np.random.uniform()
        if(coin<=alpha):
            robust_coin = True

        #print (robust_coin, alpha)
        if(robust_coin):
            with torch.no_grad(): 
                out = model(Variable(X))
                ce = nn.CrossEntropyLoss()(out, Variable(y))
                err = (out.max(1)[1] != y).float().sum()  / X.size(0)

            r = np.random.randint(low=0, high=y.shape[0], size=k)

            opt.zero_grad()

            out = model(Variable(X))
            ce = nn.CrossEntropyLoss()(out, Variable(y))
            err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)


            for ri in r:
                #print ("ri:", ri)
                #robust_ce, robust_err, temp_out, mask = sym_interval_analyze(model, epsilon, 
                #             Variable(X[ri:ri+1]), Variable(y[ri:ri+1]), use_cuda)
                robust_ce, robust_err = robust_loss(model, epsilon, 
                                                     Variable(X[ri:ri+1]), Variable(y[ri:ri+1]), 
                                                     **kwargs)
                #mask is the ranges for each layer hidden nodes.

                loss = (1.0/float(len(r)))*(interval_weight*robust_ce + ce)
                #print(ri, loss)

                loss.backward(retain_graph=True)

            if clip_grad: 
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            # measure accuracy and record loss
            losses.update(ce.item(), X.size(0))
            errors.update(err.item(), X.size(0))
            robust_losses.update(robust_ce.detach().item(), X.size(0))
            robust_errors.update(robust_err, X.size(0))

            # measure elapsed time
            batch_time.update(time.time()-end)
            end = time.time()

        else:
            out = model(Variable(X))
            ce = nn.CrossEntropyLoss()(out, Variable(y))
            err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
            loss = ce

            # measure accuracy and record loss
            losses.update(ce.item(), X.size(0))
            errors.update(err.item(), X.size(0))

            opt.zero_grad()
            loss.backward()

            if clip_grad: 
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            opt.step()
        m.set_masks(masks)# apply masks

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        if(robust_coin):
            if verbose and (i % verbose == 0): 
                endline = '\n' if i % verbose == 0 else '\r'
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                      'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                       0, i, len(loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, errors=errors, 
                       rloss = robust_losses, rerrors = robust_errors), 
                       end=endline)

            del X, y, robust_ce, out, ce, err, robust_err
        else:
            if verbose and i % verbose == 0: 
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                       0, i, len(loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, errors=errors))

            del X, y, out, ce, err

    print('')
    torch.cuda.empty_cache()


def test_vra(model, loader, args, n_steps=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()
    end = time.time()

    torch.set_grad_enabled(False)

    steps = 1
    for X,y in loader:
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        m = y.shape[0]
        rc = 0.0
        re = 0.0
        for r in range(m):
            robust_ce, robust_err, _, _ = sym_interval_analyze(model, args.epsilon, 
                                                                X[r:r+1], y[r:r+1], 
                                                                use_cuda=True, verbose=False)
            rc+=robust_ce
            re+=robust_err
        robust_ce = rc/float(m)
        robust_err = re/float(m)

        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        batch_time.update(time.time()-end)
        end = time.time()

        del X, y, robust_ce, out, ce
        if n_steps > 0 and steps==n_steps:
            break
        steps += 1

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    rerr = robust_errors.avg
    err = errors.avg
    print(' ({}) VRA {rerror:.3f}\t'
          'ACC {error:.3f}'
          .format(m, rerror=(1.0-rerr)*100, error=(1.0-err)*100))
    print('')
    return rerr
