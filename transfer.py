"""
This repo is based on https://github.com/wanglouis49/pytorch-weights_pruning code.
"""
import os 

import torch
import numpy as np
import time
import argparse
import random
from train import train_model, load_model
from utils.utils import Logger, load_cifar_dataset, set_seed
from utils.utils_transfer_attacks import transfer_madry

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


def eval_model(target_model, source_model, loader_test, args):
    target_net = torch.nn.DataParallel(target_model.model).cuda()
    source_net = torch.nn.DataParallel(source_model.model).cuda()
    target_net.load_state_dict(torch.load("trained_models_new/cifar/wideresnet/training/madry/train_madry_lr0.1_e100_nn.pth"))
    source_net.load_state_dict(torch.load("trained_models_new/cifar/wideresnet/fgsm/pruned1_epoch100_r80/init_pure/init_enhance100_madry_m0.05_nn_0.05warmup0.1_e150_nn.pth"))

    args.eps_step = 2./255.
    args.attack_iter = 10
    print("Test with PGD")
    transfer_madry(target_model, source_model, loader_test, args, n_steps=100)


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

    target_m = load_model_type(args)
    source_m = load_model_type(args)

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
    
    mask = np.load(args.mask_path, allow_pickle=True)

    source_m.set_masks(mask, transfer=args.transfer)
    target_m.set_masks(mask, transfer=args.transfer)

    eval_model(target_m, source_m, loader_test, args)


    