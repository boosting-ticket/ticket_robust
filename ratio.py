import os 

import torch
import numpy as np
import time
import argparse
from pruning.utils import prune_rate
from pruning.methods import filter_prune, weight_prune

from utils.utils_attacks import test_madry, madry_train_one_epoch
from utils.utils_trades import trades_train_one_epoch
from utils.utils_pruning import get_init_masks
from utils.utils_training import train_one_epoch, train_one_epoch_l1, test
from utils.utils_mixtrain import test_vra, mixtrain_parallel_one_epoch
from utils.utils_IBP import test_ibp_vra, ibp_one_epoch
from utils.utils import Logger, load_cifar_dataset, set_seed
from utils.utils_noise import train_one_noise_epoch, test_noise
from train import train_model, load_model, load_model_type
from config import argparse_config

import random
import sys

from torchsummary import summary


def path_config(args):
    args.model_path = os.path.join(args.model_path + args.dataset + "/")
    path = os.path.join(args.model_path + args.model_type + "/")

    if args.create_init:
        
        args.init_path = os.path.join(path + "init/")

        if args.create_init and (not os.path.exists(args.init_path)):
            os.makedirs(args.init_path)
            print("making dir:", args.init_path)

        args.log_path = None
        if args.init_type == "pure":
            args.init_path = os.path.join(args.init_path + args.init_type + "_"\
                    + args.model_type + "_init" + ".pth")
        else:
             args.init_path = os.path.join(args.init_path + args.init_type + "_"\
                    + args.model_type + "_init" + str(args.init_step) + ".pth")
        
    else:
        if args.model_name is None:
            args.model_name = "pruned"
            if args.finetune_method == "noise":
                args.model_name = "pruned_sd"+str(args.noise_sd)
            if args.finetune_method == "trades":
                args.model_name = "pruned_beta"+str(args.trades_beta)

        if not args.norm:
            args.model_name = args.model_name + "_nn"

        if not os.path.exists(path + "last/"):
            os.makedirs(path + "last/")
            print("making dir:", path + "last/")

        if args.finetune_method == "nat":
            args.last_model_path = os.path.join(path + "last/" + args.init_type + "_" + args.model_name + ".pth")
        else:
            args.last_model_path = os.path.join(path + "last/" + args.init_type + "_" + args.finetune_method + str(args.train_epochs) + "_" + args.model_name + ".pth")
        
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
                    "/pruned"+str(args.n_pruning_steps)+"_epoch" + \
                    str(args.train_epochs) + "_r" + str(args.max_pruning_ratio) +\
                    "/init_" + args.init_type\
                    + "_" + str(args.init_step) + "/")

        if not os.path.exists(path):
            os.makedirs(path)
            print("making dir:", path)

        args.model_path = os.path.join(path + args.model_name + ".pth")
        args.mask_path = os.path.join(path + args.model_name + "_mask_r" + str(args.max_pruning_ratio) + ".npy")
        args.log_path = os.path.join(path + args.model_name + ".log")
        args.results_path = os.path.join(path + args.model_name + "_result.npy")
    return args


def create_pruning_steps(args, type='decreasing'):
    """
    Create pruning schedule with args.n_pruning_steps and args.max_pruning_ratio.
    type: uniform: increase the pruning ratio with a unform step size.
          decreasing: increase the pruning ration with a decreasing step size.
    """
    assert type in ['uniform', 'decreasing']
    if type == 'uniform':
        steps = np.linspace(0, args.max_pruning_ratio, args.n_pruning_steps)
    if type == 'decreasing':
        t = np.array([i/(i+10) for i in range(1, args.n_pruning_steps+1)])
        steps = t/np.max(t)*args.max_pruning_ratio
    return steps



def update_last_enhance_method(args):
    if args.enhance_method is not None:
        args.train_epochs = 60
        args.finetune_method = args.enhance_method
        args.eps_step = 2./255.
        args.attack_iter = 10
        print("train_epoch is updated to be 30 and finetune_method is updated to " + args.enhance_method)


def ticket_pruning_with_finetuning(m, args, loader_train, loader_test):
    
    net = m.model

    steps = create_pruning_steps(args, type='decreasing')

    mask = get_init_masks(net)

    ratio = steps[-1]
    print()
    if args.parallel:
        print("Model parallel!")
        m.model = torch.nn.DataParallel(m.model).cuda()
        net = torch.load(args.last_model_path)
    else:    
        load_model(net, args.last_model_path)
    
    print("last pruned model before enhance loaded from", args.last_model_path)

    print("Pruning ratio = {:.3f}".format(ratio))
    '''
    # Calculate the new mask from the finetuned network
    '''
    if args.prune_method == "unstructured":
        # Unstructured pruning
        m.set_masks(mask)
        mask = weight_prune(m, ratio, verbose=False)
    if args.prune_method == "structured":
        # Structured pruning
        m.set_masks(mask)
        mask = filter_prune(m, mask, ratio, verbose=False)

    np.save(args.mask_path, np.array(mask))
    print("mask saved in", args.mask_path)
    exit()

    


# python ticket.py vgg16 pure 
# --finetune_method nat (finetune method)
# --enhance_method None (=finetune_method, the method used for last step of finetune)
# --gpu 0 (=0)
# --learning_rate (=0.01, the lr used for enhancement)
# --n_pruning_steps (=1, numebr of iterative pruning steps)
# --train_epochs (=60, the epoch used for enhancement)
# --model_name (=default, will save to model_name.log/pth)
# --create_init (=False, just save the model if True)
# --init_step (=0 if pure, number of rewinding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = argparse_config(parser)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.seed is None:
        args.seed = 7

    set_seed(args.seed)

    loader_train, loader_valid, loader_test = load_cifar_dataset(args, data_dir='./data')
    
    assert args.finetune_method in ["nat", "nat_l1", "madry", "mixtrain", "fgsm",\
                        "naive", "sym", "noise", "trades", "trades_fgsm"],\
                        "no such finetuning method!"
    assert args.enhance_method in [None, "nat", "madry", "mixtrain", "fgsm",\
                        "naive", "sym", "noise", "trades", "trades_fgsm"],\
                        "no such finetuning method!"
    assert args.train_method in ["nat", "nat_l1", "madry", "mixtrain", "fgsm",\
                        "naive", "sym", "noise", "trades", "trades_fgsm"],\
                        "no such train method!"
    assert args.prune_method in ["unstructured", "structured"]
    assert args.init_type in ["nat", "madry", "pure", "trades"]
    assert args.init_step > 0, "please use init_type=pure if init_step is 0"

    m = load_model_type(args)
    net = m.model

    args = path_config(args)
    
    if torch.cuda.is_available():
        print('CUDA enabled.')
        net = net.cuda()

    for k in args.__dict__:
        print(k, ":", args.__dict__[k])

    print("config:")

    #print("Start ticket pruning on model", args.model_path + args.train_method + "_vgg16_init100.pth")
    print("Start ticket pruning on model", args.init_path)
    print("Pruning method:", args.prune_method)
    print("Finetune method:", args.finetune_method)
    print("Pruned model will be saved in", args.model_path)
    print("Final mask will be saved in", args.mask_path)
    print("Log will be saved in", args.log_path)
    print()

    # Main function for ticket pruning
    results = ticket_pruning_with_finetuning(m, args, loader_train, loader_test)
    print(results)

    #np.save(args.results_path, np.array(results))
    #torch.save(m.model.state_dict(), args.model_path)
    
    


        




