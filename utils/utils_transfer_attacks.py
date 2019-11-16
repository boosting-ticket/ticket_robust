import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np
import time 


def cal_loss(y_out, y_true, targeted):
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(y_out, y_true)
    if targeted:
        return loss_cal
    else:
        return -1*loss_cal


def generate_target_label_tensor(true_label, args):
    t = torch.floor(10*torch.rand(true_label.shape)).type(torch.int64)
    m = t == true_label
    t[m] = (t[m]+ torch.ceil(9*torch.rand(t[m].shape)).type(torch.int64)) % args.n_classes
    return t


def pgd_attack(model, image_tensor, img_variable, tar_label_variable,
               n_steps, eps_max, eps_step, clip_min, clip_max, targeted):
    """
    image_tensor: tensor which holds the clean images. 
    img_variable: Corresponding pytorch variable for image_tensor.
    tar_label_variable: Assuming targeted attack, this variable holds the targeted labels. 
    n_steps: number of attack iterations. 
    eps_max: maximum l_inf attack perturbations. 
    eps_step: l_inf attack perturbation per step
    """
    model.eval()
    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)
        output = model.forward(img_variable)
        loss_cal = cal_loss(output, tar_label_variable, targeted)
        loss_cal.backward()
        x_grad = -1 * eps_step * torch.sign(img_variable.grad.data)
        adv_temp = img_variable.data + x_grad
        total_grad = adv_temp - image_tensor
        total_grad = torch.clamp(total_grad, -eps_max, eps_max)
        x_adv = image_tensor + total_grad
        x_adv = torch.clamp(torch.clamp(
            x_adv-image_tensor, -1*eps_max, eps_max)+image_tensor, clip_min, clip_max)
        img_variable.data = x_adv
    #print("peturbation= {}".format(
    #    np.max(np.abs(np.array(x_adv)-np.array(image_tensor)))))
    return img_variable

      

def transfer_madry(target_model, source_model, loader, args, n_steps=0):
    """
    n_steps (int): Number of batches for evaluation.
    """
    target_model.eval()
    source_model.eval()
    num_correct, num_correct_adv, num_samples = 0, 0, 0
    steps = 1

    if args.targeted:
        print("Evaluating with pgd attack => targeted = {}, eps = {:.3f},"\
            "number of iterations = {}".format(args.targeted,\
            args.epsilon, args.attack_iter))
    else:
        print("Evaluating with pgd untargeted attack,"\
            "eps=%.6f, iters=%d, attack_steps=%f"%\
            (args.epsilon, args.attack_iter, args.eps_step))

    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        if args.targeted:
            y_target = generate_target_label_tensor(
                               y_var.cpu(), args).cuda()
        else:
            y_target = y_var
        adv_x = pgd_attack(source_model,
                           x,
                           x_var,
                           y_target,
                           args.attack_iter,
                           args.epsilon,
                           args.eps_step,
                           args.clip_min,
                           args.clip_max,
                           args.targeted)
        scores_adv = target_model(adv_x)
        _, preds_adv = scores_adv.data.max(1)
        num_correct_adv += (preds_adv == y).sum()
        num_samples += len(preds_adv)
        if n_steps > 0 and steps==n_steps:
            break
        steps += 1
        
    acc_adv = float(num_correct_adv) / num_samples
    print('Adversarial accuracy: {:.2f}% ({}/{})'.format(
        100.*acc_adv,
        num_correct_adv,
        num_samples,
    ))

    return acc_adv
        

def test_fgsm_noise(model, loader, args, n_steps=0):
    """
    n_steps (int): Number of batches for evaluation.
    """
    model.eval()
    num_correct, num_correct_adv, num_samples = 0, 0, 0
    steps = 1
    noise = args.epsilon / 2.
    args.epsilon = args.epsilon / 2.

    if args.targeted:
        print("Evaluating with pgd attack => targeted = {}, eps = {:.3f},"\
            "number of iterations = {}".format(args.targeted,\
            args.epsilon, args.attack_iter))
    else:
        print("Evaluating with pgd untargeted attack, eps=%.6f,"\
            "noise=%.6f, iters=%d, attack_steps=%f"%(args.epsilon,\
            noise, args.attack_iter, args.eps_step))

    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        noise_sign = (torch.randint(0, 2, x.shape)-1).float().cuda()
        x_var = Variable(x, requires_grad= True)
        y_var = Variable(y, requires_grad= False)
        if args.targeted:
            y_target = generate_target_label_tensor(
                               y_var.cpu(), args).cuda()
        else:
            y_target = y_var
        adv_x = pgd_attack(model,
                           x + noise * noise_sign,
                           x_var + noise * noise_sign,
                           y_target,
                           args.attack_iter,
                           args.epsilon,
                           args.eps_step,
                           args.clip_min,
                           args.clip_max,
                           args.targeted)
        scores = model(x.cuda()) 
        _, preds = scores.data.max(1)
        scores_adv = model(adv_x)
        _, preds_adv = scores_adv.data.max(1)
        num_correct += (preds == y).sum()
        num_correct_adv += (preds_adv == y).sum()
        num_samples += len(preds)
        if n_steps > 0 and steps==n_steps:
            break
        steps += 1

    acc = float(num_correct) / num_samples
    acc_adv = float(num_correct_adv) / num_samples
    print('Clean accuracy: {:.2f}% ({}/{})'.format(
        100.*acc,
        num_correct,
        num_samples,
    ))
    print('Adversarial accuracy: {:.2f}% ({}/{})'.format(
        100.*acc_adv,
        num_correct_adv,
        num_samples,
    ))

    return acc, acc_adv