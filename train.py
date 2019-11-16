import os 

import torch
import torch.nn as nn
import numpy as np
import time
from utils.utils_attacks import test_madry, madry_train_one_epoch
from utils.utils_trades import trades_train_one_epoch
from utils.utils_pruning import get_init_masks
from utils.utils_training import train_one_epoch, train_one_epoch_l1, test
from utils.utils_mixtrain import test_vra, mixtrain_parallel_one_epoch
from utils.utils_IBP import test_ibp_vra, ibp_one_epoch
from utils.utils_noise import train_one_noise_epoch, test_noise
from utils.utils import Logger, load_cifar_dataset, set_seed

from models import cifar_model_large, vgg, vgg_no_bn, cifar_model_small, cifar_conv2, cifar_conv4, cifar_conv4_bn_max2
from wideresnet import WideResNet
from resnet import ResNet18

import argparse
from config import argparse_config
import random
import sys
import math
def load_model(net, model_path):
    print("model loading from " + model_path)
    net.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available:
        net = net.cuda()


def load_model_type(args):
    '''
    assert args.model_type in["vgg16", "large", "small", "conv2", "wideresnet",\
                        "wideresnet_28_10", "wideresnet_28_5", "resnet18", "vgg11",
                        "vgg8s", "vgg8", "vgg7", "conv2_4", "conv2_16", "conv4", "conv4_2",
                        "vgg8_no_bn", "conv4_bn_max2"]
    '''

    if args.model_type == "vgg16":
        m = vgg(dataset=args.dataset)
    elif args.model_type == "vgg11":
        m = vgg(depth=11)
    elif args.model_type == "vgg8s":
        m = vgg(depth='8s')
    elif args.model_type == "vgg8":
        m = vgg(depth=8)
    elif args.model_type == "vgg7":
        m = vgg(depth=7)
    elif args.model_type == "vgg7_avg4":
        m = vgg(depth=7)
    elif args.model_type == "vgg8_no_bn":
        m = vgg_no_bn(depth=8)
    elif args.model_type == "conv2":
        m = cifar_conv2()
    elif args.model_type == "conv2_4":
        m = cifar_conv2(depth=4)
    elif args.model_type == "conv2_16":
        m = cifar_conv2(depth=16)
    elif args.model_type == "conv4":
        m = cifar_conv4(depth=1)
    elif args.model_type == "conv4_bn_max2":
        m = cifar_conv4_bn_max2(depth=1)
    elif args.model_type == "conv4_2":
        m = cifar_conv4(depth=2)
    elif args.model_type == "large":
        m = cifar_model_large(width=args.model_width)
        args.model_type = args.model_type + "_w" + str(args.model_width)
    elif args.model_type == "small":
        m = cifar_model_small(width=args.model_width)
        args.model_type = args.model_type + "_w" + str(args.model_width)

    elif args.model_type == "wideresnet":
        m = WideResNet()
    elif args.model_type == "wideresnet_34_8":
        m = WideResNet(depth=34, widen_factor=8)
    elif args.model_type == "wideresnet_34_5":
        m = WideResNet(depth=34, widen_factor=5)
    elif args.model_type == "wideresnet_34_2":
        m = WideResNet(depth=34, widen_factor=2)
    elif args.model_type == "wideresnet_34_1":
        m = WideResNet(depth=34, widen_factor=1)

    elif args.model_type == "wideresnet_28_10":
        m = WideResNet(depth=28, widen_factor=10)
    elif args.model_type == "wideresnet_22_10":
        m = WideResNet(depth=22, widen_factor=10)
    elif args.model_type == "wideresnet_16_10":
        m = WideResNet(depth=16, widen_factor=10)
    elif args.model_type == "wideresnet_10_10":
        m = WideResNet(depth=10, widen_factor=10)

    elif args.model_type == "resnet18":
        m = ResNet18()

    else:
        print("no such model type")

    if torch.cuda.is_available:
        m.model = m.model.cuda()

    return m


class early_stop_evaluation():
    def __init__(self, max_size=10):
        self.loss_valid_queue = [100000]
        self.max_size = max_size
        self.save = False

    def check_early_stop(self, loss_valid):
        if loss_valid >= self.loss_valid_queue[0] - 1e-5:
            self.save = False
            self.loss_valid_queue.append(loss_valid)
            if len(self.loss_valid_queue) == self.max_size:
                return True
        else:
            self.loss_valid_queue = [loss_valid]
            self.save = True
            return False

    def check_save_best(self):
        if self.save:
            return True
        else:
            return False


def train_model(m, mask, loader_train, loader_valid, loader_test, args, train_type, verbose=False):

    net = m.model.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.optm == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9, 
                                    weight_decay=args.weight_decay)
    elif args.optm == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    else:
        print("no such optimizer!")

    early_check = early_stop_evaluation(max_size=args.early_stop)

    best_acc = -1
    acc = -1
    best_pgd_acc = -1
    pgd_acc = -1
    best_vra = -1
    vra = -1
    start_time = time.time()

    for t in range(args.resume, args.train_epochs):
        current_time = time.time()-start_time
        if t==args.resume: 
            print("Epoch [%d/%d]"%(t, args.train_epochs))
        else:
            print("Epoch [%d/%d], Passed time:[%.3f/%.3f]"%(t, args.train_epochs,\
                        (current_time)/(t-args.resume), current_time))
        #optimizer.param_groups[0]["lr"] = set_lr(args, t)
        set_lr(args, t, optimizer)
        print("learning rate:", optimizer.param_groups[0]["lr"])

        '''
        # if args.is_training: train_type is args.train_method
        # else: train_type is args.finetune_method
        # therefore we have to saperate it
        '''
        if train_type == "nat":
            train_one_epoch(m, mask, criterion, optimizer, loader_train, args, verbose=args.verbose)
            print("Valid Test with nat")
            acc, ce = test(net, loader_valid, criterion)

            if args.early_stop > 0:
                if early_check.check_early_stop(ce):
                    print("Early stop at ", early_check.loss_valid_queue, "best model loaded from", args.model_path)
                    load_model(net, args.model_path)
                    print("Test with test set:")
                    best_acc, best_ce = test(net, loader_test, criterion)
                    return best_acc, best_pgd_acc

        elif train_type == "nat_l1":
            train_one_epoch_l1(m, mask, criterion, optimizer, loader_train, args, verbose=args.verbose)
            print("Valid Test with nat")
            acc, ce = test(net, loader_valid, criterion)

            if args.early_stop > 0:
                if early_check.check_early_stop(ce):
                    print("Early stop at ", early_check.loss_valid_queue, "best model loaded from", args.model_path)
                    load_model(net, args.model_path)
                    print("Test with test set:")
                    best_acc, best_ce = test(net, loader_test, criterion)
                    return best_acc, best_pgd_acc

        elif train_type == "madry":
            args.eps_step = 2./255.
            args.attack_iter = 10
            print("Train with PGD")
            madry_train_one_epoch(m, mask, criterion, optimizer, loader_train, args, verbose=args.verbose)
            print("Valid Test with PGD")
            acc, pgd_acc = test_madry(net, loader_valid, args)

        elif train_type == "trades":
            args.eps_step = 2./255.
            args.attack_iter = 10
            print("Train with Trades")
            trades_train_one_epoch(m, mask, optimizer, loader_train, args, verbose=args.verbose)
            print("Valid Test with PGD")
            acc, pgd_acc = test_madry(net, loader_valid, args)

            if pgd_acc > best_pgd_acc:
                best_pgd_acc = pgd_acc
                torch.save(m.model.state_dict(), args.model_path)
                print("best model saved to ", args.model_path, "with acc", acc, "and pgd_acc", pgd_acc)

        elif train_type == "trades_fgsm":
            args.eps_step = 8./255.
            args.attack_iter = 1
            print("Train with Trades FGSM")
            trades_train_one_epoch(m, mask, optimizer, loader_train, args, verbose=args.verbose)

            args.eps_step = 2./255.
            args.attack_iter = 10
            if t % 10 == 9:
                print("Valid Test with PGD")
                acc, pgd_acc = test_madry(net, loader_valid, args)
                args.eps_step = 8./255.
                args.attack_iter = 1
            
        elif train_type == "fgsm":
            args.eps_step = 8./255.
            args.attack_iter = 1
            print("Train with FGSM")
            madry_train_one_epoch(m, mask, criterion, optimizer, loader_train, args, verbose=args.verbose)
            if t % 20 == 19:
                print("Valid Test with FGSM")
                acc, pgd_acc = test_madry(net, loader_valid, args)
                
                args.eps_step = 2./255.
                args.attack_iter = 10
                print("Test with PGD")
                acc, pgd_acc = test_madry(net, loader_valid, args)
                args.eps_step = 8./255.
                args.attack_iter = 1
            
        elif train_type == "mixtrain":
            mixtrain_parallel_one_epoch(loader_train, m, mask, optimizer, 
                    args.epsilon, interval_weight=args.ft_interval_weight, verbose=args.verbose, 
                    bounded_input=False, clip_grad=1)

        elif train_type == "naive":
            print(t, args.schedule_length, min(float(t)/float(args.schedule_length), 1.))
            epsilon = args.epsilon * min(float(t)/float(args.schedule_length), 1.)
            ibp_one_epoch(loader_train, m, mask, optimizer, 
                    epsilon, interval_weight=args.ft_interval_weight, verbose=args.verbose, 
                    clip_grad=1)
            print("Valid Test with naive")
            vra = test_ibp_vra(m.model, loader_valid, args)
        elif train_type == 'noise':
            train_one_noise_epoch(m, mask, criterion, args.noise_sd, optimizer, loader_train, verbose=args.verbose)
            print("Valid Test with noise")
            acc = test_noise(net, args.noise_sd, loader_valid)
        else:
            print("no such traininig method!")
            exit()

        
        if early_check.check_save_best() and not args.create_init:

            #torch.save(net.state_dict(), args.model_path)
            print("Best model so far saved in", args.model_path)
            print("Test on test set:")
            if train_type in ["nat", "nat_l1"]:
                best_acc, best_ce = test(net, loader_test, criterion)
            else:
                acc, pgd_acc = test_madry(net, loader_test, args)

    #torch.save(net.state_dict(), args.model_path)
    print("Training done, model saved in", args.model_path)
    print("Test on test set:")
    if train_type == "nat":
        best_acc, best_ce = test(net, loader_test, criterion)
    else:
        acc, pgd_acc = test_madry(net, loader_test, args)
        
    return best_acc, best_pgd_acc


def set_lr(args, t, optimizer):
    if args.warmup:
        for param_group in optimizer.param_groups:
            if args.model_type == "vgg16":
                if t<10:
                    # no warmup for finding mask should be 0.01
                    # warmup for enhancement should be from 0.01 to 0.1
                    param_group['lr'] = 0.1*(t+1)*args.learning_rate
            elif args.model_type == "resnet18":
                if t<11:
                    # warmup for finding mask should be from 0.01 to 0.1(best)/0.005 to 0.05
                    # no warmup for finding mask should be 0.05 compare to 0.1
                    # warmup for enhancement should be from 0.05 to 0.1
                    param_group['lr'] = t*(args.learning_rate - 0.05)/10. + 0.05
            elif "wideresnet" in args.model_type:
                if t<11:
                    # warmup for finding mask should be from 0.01 to 0.1(best)/0.005 to 0.05
                    # no warmup for finding mask should be 0.05 compare to 0.1
                    # warmup for enhancement should be from 0.05 to 0.1
                    #param_group['lr'] = 0.1*(t+1)*args.learning_rate
                    param_group['lr'] = t*(args.learning_rate - 0.05)/10. + 0.05

    if t in [math.ceil(args.train_epochs*0.5), math.ceil(args.train_epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate * (0.6 ** ((max((t-args.schedule_length), 0) // 5)))
    '''


def path_config(args):
    args.model_path = os.path.join(args.model_path + args.dataset + "/")
    path = os.path.join(args.model_path + args.model_type + "/")

    if args.model_name is None:
        args.model_name = "train"
        if args.train_method == "noise":
            args.model_name = "train_sd"+str(args.noise_sd)
        if args.train_method == "trades":
            args.model_name = "train_beta"+str(args.trades_beta)

    if not args.norm:
        args.model_name = args.model_name + "_nn"

    path = os.path.join(path + "training/" + args.train_method + "/")

    if not os.path.exists(path):
        os.makedirs(path)
        print("making dir:", path)
    args.model_path = os.path.join(path + args.model_name + ".pth")
    args.log_path = os.path.join(path + args.model_name + ".log")
    return args


# python train.py vgg16 pure
# --train_method nat (train method)
# --gpu 0 (=0)
# --learning_rate (=0.01, the lr used for enhancement)
# --train_epochs (=train_epochs, the epoch used for enhancement)
# --model_name (=default, will save to model_name.log/pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = argparse_config(parser)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.seed is None:
        args.seed = 7

    set_seed(args.seed)

    loader_train, loader_valid, loader_test = load_cifar_dataset(args, data_dir='./data')
    
    assert args.train_method in ["nat", "madry", "mixtrain", "fgsm",\
                        "naive", "sym", "noise", "trades", "trades_fgsm"],\
                        "no such train method!"
    assert args.init_type in ["nat", "madry", "pure", "trades"]
    assert args.init_step > 0, "please use init_type=pure if init_step is 0"

    m = load_model_type(args)
    net = m.model

    args = path_config(args)

    if args.log_path is not None:
        log = open(args.log_path, "w")
        sys.stdout = Logger(log)
    
    if torch.cuda.is_available():
        print('CUDA enabled.')
        net = net.cuda()

    for k in args.__dict__:
        print(k, ":", args.__dict__[k])

    print("config:")
    print("training method:", args.train_method)
    print("model will be saved in:", args.model_path)
    print("Log will be saved in", args.log_path)
    if args.resume !=0:
        print("Resume from epoch", args.resume)
        log = open(args.log_path, "a")
        sys.stdout = Logger(log)
        load_model(m.model, args.model_path)
        print("model resumed from", args.model_path)
    
    # Just regular training with args.train_method
    if args.parallel:
        print("Model parallel!")
        m.model = nn.DataParallel(m.model).cuda()
    train_model(m, get_init_masks(net), loader_train, loader_valid, loader_test,\
            args, train_type=args.train_method, verbose=False) 
    exit()
    


        
