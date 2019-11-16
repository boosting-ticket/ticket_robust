import os 

import torch
import numpy as np
import time
import argparse
import random
from train import train_model, load_model
from utils.utils import Logger, load_cifar_dataset, set_seed

from train import train_model, load_model, load_model_type
from config import argparse_config


def path_config(path):
    args.model_path = os.path.join(args.model_path + args.dataset + "/")
    path = os.path.join(args.model_path + args.model_type + "/")

    if args.model_name is None:
        if args.init:
            args.model_name = "init_enhance_" + args.enhance_method
        else:
            args.model_name = "seed9_enhance_" + args.enhance_method

        if args.eval:
            args.model_name = args.model_name+"_eval"

    if not args.norm:
        args.model_name = args.model_name + "_nn"

    if args.init_type == "pure":
        args.init_path = os.path.join(path + "init/" + args.init_type + "_"\
                + args.model_type + "_init" + ".pth")

        path = os.path.join(path + args.finetune_method +\
                "/pruned"+str(args.n_pruning_steps)+"_epoch" +\
                str(args.train_epochs) + "_r" + str(args.max_pruning_ratio) +\
                "/init_" + args.init_type\
                + "/")
    else:
        args.init_path = os.path.join(path + "init/" + args.init_type + "_"\
                + args.model_type + "_init" + str(args.init_step) + ".pth")

        path = os.path.join(path + args.finetune_method +\
                "/pruned"+str(args.n_pruning_steps)+"_epoch" +\
                str(args.train_epochs) + "_r" + str(args.max_pruning_ratio) +\
                "/init_" + args.init_type\
                + "_" + str(args.init_step) + "/")

    if not os.path.exists(path):
        os.makedirs(path)
        print("making dir:", path)

    args.model_path = os.path.join(path + args.model_name + ".pth")
    if args.mask_name is None:
        args.mask_path = os.path.join(path + "pruned" + "_mask_r" + str(args.max_pruning_ratio) + ".npy")
    else:
        args.mask_path = os.path.join(path + args.mask_name + ".npy")
    args.log_path = os.path.join(path + args.model_name + ".log")
    return args


def eval_model(m, loader_test, args, test_type):

    net = m.model.cuda()

    if test_type == "nat":
        acc = test(net, loader_test)

    elif test_type == "madry":
        args.eps_step = 2./255.
        args.attack_iter = 10
        print("Test with PGD")
        pgd_acc = test_madry(net, loader_test, args, n_steps=100)

    elif test_type == "fgsm":
        args.eps_step = 8./255.
        args.attack_iter = 1
        print("Test with FGSM")
        pgd_acc = test_madry(net, loader_test, args, n_steps=100)
        
        args.eps_step = 2./255.
        args.attack_iter = 10
        print("Test with PGD")
        pgd_acc = test_madry(net, loader_test, args, n_steps=100)
        args.eps_step = 8./255.
        args.attack_iter = 1

    else:
        print("no such traininig method!")

    return net, mask


# python enhance.py vgg16 pure 
# --finetune_method nat --n_pruning_steps 1 --train_epochs 60 (locate the finetune model and mask)
# --mask_name (="pruned_mask_r80", customized which mask to use in pruned1_epoch60)
# --enhance_method nat (=finetune_method if None)
# --gpu 0 (=0)
# --init (=False, whether to load the init model and set seed to original 7 instead of new seed 9)
# --model_name (=default, will save to model_name.log/pth)
# --enhance_epochs (=train_epochs, the epoch used for enhancement)
# --enhance_learning_rate (=learning_rate, the lr used for enhancement)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = argparse_config(parser)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.seed is None:
        if args.init: args.seed = 7
        else: args.seed = 9

    set_seed(args.seed)
    
    loader_train, loader_valid, loader_test = load_cifar_dataset(args, data_dir='./data')

    if args.enhance_method is None:
        args.enhance_method = args.finetune_method

    assert args.finetune_method in ["nat", "madry", "mixtrain", "fgsm",\
                        "naive", "sym", "noise", "trades", "trades_fgsm"],\
                        "no such finetuning method!"
    assert args.enhance_method in [None, "nat", "madry", "mixtrain", "fgsm",\
                        "naive", "sym", "noise", "trades", "trades_fgsm"],\
                        "no such finetuning method!"
    assert args.init_type in ["nat", "madry", "pure", "trades"]
    assert args.init_step > 0, "please use init_type=pure if init_step is 0"

    m = load_model_type(args)
    net = m.model

    args = path_config(args)

    if args.enhance_learning_rate is not None:
        args.learning_rate = args.enhance_learning_rate

    if args.enhance_epochs is not None:
        args.train_epochs = args.enhance_epochs

    import sys
    log = open(args.log_path, "w")
    sys.stdout = Logger(log)

    for k in args.__dict__:
        print(k, ":", args.__dict__[k])

    if torch.cuda.is_available():
        print('CUDA enabled.')
        net = net.cuda()

    print("Enhance training config:")
    print("Enhance training method:", args.enhance_method)
    print("model will be saved in:", args.model_path)
    print("Init model is:", args.init_path)
    print("Init mask used from:", args.mask_path)
    print("Log will be saved in", args.log_path)
    print("Random seed is:", args.seed)
    print()

    if args.init:
        if not args.transfer:
            load_model(net, args.init_path)
        else:
            from models import vgg
            m_orig = vgg()
            load_model(m_orig.model, args.init_path)
            m.transfer_model(m_orig)

    mask = np.load(args.mask_path, allow_pickle=True)

    m.set_masks(mask, transfer=args.transfer)

    if args.eval:
        eval_model(m, loader_test, args, test_type=args.test_method)
    else:
        if args.parallel:
            print("Model parallel!")
            m.model = torch.nn.DataParallel(m.model).cuda()
        train_model(m, mask, loader_train, loader_valid, loader_test,\
            args, train_type=args.enhance_method, verbose=False) 

    