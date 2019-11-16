import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np
import time 


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        nature_logits = F.softmax(model(x_natural), dim=1)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       nature_logits)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            for idx_batch in range(batch_size):
                grad_idx = grad[idx_batch]
                grad_idx_norm = l2_norm(grad_idx)
                grad_idx /= (grad_idx_norm + 1e-8)
                x_adv[idx_batch] = x_adv[idx_batch].detach() + step_size * grad_idx
                eta_x_adv = x_adv[idx_batch] - x_natural[idx_batch]
                norm_eta = l2_norm(eta_x_adv)
                if norm_eta > epsilon:
                    eta_x_adv = eta_x_adv * epsilon / l2_norm(eta_x_adv)
                x_adv[idx_batch] = x_natural[idx_batch] + eta_x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, x_adv

            
def trades_train_one_epoch(m, masks, optimizer, loader_train, args, verbose=True):
    model = m.model
    model.train()
    for t, (x, y) in enumerate(loader_train):
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad=True)
        y_var = Variable(y, requires_grad=False)
        
        loss, adv_x = trades_loss(model,
                           x_var,
                           y_var,
                           optimizer,
                           args.eps_step,
                           args.epsilon,
                           args.attack_iter,
                           beta=args.trades_beta)
        
        scores = model(adv_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        m.set_masks(masks) # apply masks

        if args.create_init:
            if args.init_step == 0:
                torch.save(m.model.state_dict(), args.init_path)
                print("init", args.init_step, "model saved to", args.init_path)
                exit()
            else: 
                args.init_step -= 1

        if (verbose > 0) and (t % verbose == 0):
            acc = float((scores.max(1)[1]==y).sum())/float(y.shape[0])
            print('Batch [%d/%d] training loss = %.8f, training trades_acc = %.8f' % (t, len(loader_train), loss.data, acc))
        