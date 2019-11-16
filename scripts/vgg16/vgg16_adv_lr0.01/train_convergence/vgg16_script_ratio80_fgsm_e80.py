#! /bin/bash
#18, 32, 45, 56, 65, 73, 80, 87, 92, 98

RATIO = 80
TRAIN_LR = 0.1

TRAIN_EPOCH = 80
MODEL_NAME = "train_lr0.01warmup0.1_e"+str(TRAIN_EPOCH)
MASK_NAME = MODEL_NAME+"_nn" + "_mask_r"+str(RATIO)
GPU = 0
TRAIN_METHOD = "madry"


# create init model
cmd_init = "python ticket.py vgg16 pure --gpu " + str(GPU) + " --create_init"
# create last model used for getting pruning mask for different pruning ratio
cmd0 = "python train.py vgg16 pure --norm --gpu " + str(GPU) + " --train_epochs "+str(TRAIN_EPOCH)+" --train_method "+TRAIN_METHOD+" --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME


import sys
import os
#os.system(cmd_init)
os.system(cmd0)
