import sys
from utils.data_loader import get_train_valid_loader, get_test_loader
import torch
import numpy as np

import random

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class Logger(object):
    def __init__(self, stream):
        self.stream = stream
        self.terminal = sys.stdout
    def write(self, data):
        self.terminal.write(data)
        self.stream.write(data)
        self.stream.flush()
        self.terminal.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.terminal.writelines(datas)
        self.terminal.flush()
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic=True



def load_cifar_dataset_no_validation(args, data_dir):
    # CIFAR-10 data loaders

    if args.norm:
    
        trainset = datasets.CIFAR10(root=data_dir, train=True,
                                    download=True, 
                                    transform=transforms.Compose([
                                        transforms.Pad(4),
                                        transforms.RandomCrop(32, 4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ]))
        testset = datasets.CIFAR10(root=data_dir,
                                train=False,
                                download=True, 
                                transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                               ]))
    else:
        trainset = datasets.CIFAR10(root=data_dir, train=True,
                                    download=True, 
                                    transform=transforms.Compose([
                                        transforms.Pad(4),
                                        transforms.RandomCrop(32, 4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()
                                    ]))
        testset = datasets.CIFAR10(root=data_dir,
                                train=False,
                                download=True, 
                                transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

    loader_train = torch.utils.data.DataLoader(trainset, 
                                batch_size=args.batch_size,
                                shuffle=True, worker_init_fn=np.random.seed(7),
                                num_workers=1, pin_memory=True)

    
    loader_test = torch.utils.data.DataLoader(testset, 
                                batch_size=args.test_batch_size,
                                shuffle=False, worker_init_fn=np.random.seed(7),
                                num_workers=1, pin_memory=True)
    return loader_train, loader_test, loader_test
    

def load_cifar_dataset(args, data_dir):
    if args.parallel: 
        num_workers = 50
    else: 
        num_workers = 1

    if args.finetune_method == "nat":
        loader_train, loader_valid = get_train_valid_loader(data_dir, args.batch_size, dataset=args.dataset, norm=args.norm, num_workers=num_workers)
        loader_test = get_test_loader(data_dir, args.test_batch_size, dataset=args.dataset, norm=args.norm)
        return loader_train, loader_valid, loader_test
    else:
        return load_cifar_dataset_no_validation(args, data_dir='./data')



