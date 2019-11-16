#! /bin/bash
#18, 32, 45, 56, 65, 73, 80, 87, 92, 98

RATIO = 80
TRAIN_LR = 0.1
# Need to test whether it's the best or 0.01
ENHANCE_LR = 0.1

MODEL_NAME = "train_trades6_lr" + str(TRAIN_LR)+"_e150"
MASK_NAME = MODEL_NAME+"_nn" + "_mask_r"+str(RATIO)

ENHANCE_METHOD = "trades"
FINETUNE_METHOD = "fgsm"

GPU = "2,3"

MODEL_TYPE = "wideresnet"
TRAIN_EPOCH = 150

# create init model
cmd_init = "python ticket.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --create_init"
# create last model used for getting pruning mask for different pruning ratio
cmd0 = "python train.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --train_method trades --norm --max_pruning_ratio " + str(RATIO) + " --train_epochs "+str(TRAIN_EPOCH)+" --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME


import sys
import os
#os.system(cmd_init)
os.system(cmd0)
