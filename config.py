import os 

import torch
import numpy as np
import time
import argparse
from wideresnet import WideResNet
from resnet import ResNet18

import random
import sys

def config_vgg16(args):

    if args.learning_rate is None:
        args.learning_rate = 0.01

    if args.batch_size is None:
        args.batch_size = 64

    if args.norm is None:
        args.norm = True

    if args.train_epochs is None:
        args.train_epochs = 100

    return args

def config_resnet18(args):

    if args.learning_rate is None:
        args.learning_rate = 0.05

    if args.batch_size is None:
        args.batch_size = 128

    if args.norm is None:
        args.norm = True

    if args.train_epochs is None:
        args.train_epochs = 100

    return args


def config_wideresnet(args):

    if args.learning_rate is None:
        args.learning_rate = 0.1

    if args.batch_size is None:
        args.batch_size = 128

    if args.norm is None:
        args.norm = True

    if args.train_epochs is None:
        args.train_epochs = 100
    
    args.parallel = True

    return args



def argparse_config(parser):

    parser.add_argument('model_type', type=str)
    parser.add_argument('init_type', type=str, default="pure")
    parser.add_argument('--finetune_method', default="nat")
    parser.add_argument('--enhance_method', type=str, default=None, help="keep the same as finetune method if None")
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--model_width', type=int, default=8)
    parser.add_argument('--n_pruning_steps', type=int, default=1) 
    parser.add_argument('--max_pruning_ratio', type=int, default=80)
    parser.add_argument('--train_epochs', type=int, default=None)
    parser.add_argument('--enhance_epochs', type=int, default=None)
    parser.add_argument('--prune_method', default="unstructured")
    parser.add_argument('--dataset', type=str, default="cifar")
    parser.add_argument('--noise_sd', type=float, default=1.0)
    parser.add_argument('--trades_beta', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--warmup', action="store_false", default=True)
    parser.add_argument('--parallel', action="store_true", default=False)

    parser.add_argument('--create_init', action="store_true", default=False)
    parser.add_argument('--init_step', type=int, default=1400)

    parser.add_argument('--train_method', default="nat")

    parser.add_argument('--early_stop', type=int, default=1000) 
    parser.add_argument('--norm', action="store_false", default=None) 
    parser.add_argument('--optm', type=str, default="sgd") 
    parser.add_argument('--batch_size', type=int, default=None) 
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--enhance_learning_rate', type=float, default=None)
    parser.add_argument('--schedule_length', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=8.0/255.0)
    parser.add_argument('--attack_iter', type=int, default=10)
    parser.add_argument('--eps_step', type=float, default=2.0/255)
    parser.add_argument('--targeted', type=bool, default=False)
    parser.add_argument('--clip_min', type=float, default=0)
    parser.add_argument('--clip_max', type=float, default=1.0)
    parser.add_argument('--starting_epsilon', type=float, default=0.00001)
    parser.add_argument('--interval_weight', type=float, default=0.1)
    parser.add_argument('--ft_interval_weight', type=float, default=50)
    parser.add_argument('--verbose', type=int, default=200, help="not verbose if 0")
    parser.add_argument('--resume', type=int, default=0)

    parser.add_argument('--model_path', type=str, default='./trained_models_new/')
    parser.add_argument('--last_model_path', type=str, default='./trained_models_new/')
    parser.add_argument('--mask_path', type=str, default=None)
    parser.add_argument('--mask_name', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--init_path', type=str, default=None)
    parser.add_argument('--results_path', type=str, default=None)
    parser.add_argument('--n_classes', type=int, default=10)

    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--init', action="store_true", default=False)
    parser.add_argument('--transfer', action="store_true", default=False)

    args = parser.parse_args()

    assert args.model_type in ["vgg16", "resnet18", "wideresnet"] or "wideresnet" in args.model_type

    if args.model_type == "vgg16":
        args = config_vgg16(args)

    if args.model_type == "resnet18":
        args = config_resnet18(args)

    if "wideresnet" in args.model_type:
        args = config_wideresnet(args)

    if args.transfer is None:
        if args.dataset == "cifar100":
            args.transfer = True
        else:
            args.transfer = False

    if args.transfer:
        assert args.dataset == "cifar100", "only transfer from cifar100"
        args.model_path = "./trained_models_transfer/"


    return args


        




