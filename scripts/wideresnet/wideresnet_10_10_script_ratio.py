#! /bin/bash
#18, 32, 45, 56, 65, 73, 80, 87, 92, 98

RATIO = 80
TRAIN_LR = 0.05

MODEL_NAME = "pruned_lr" + str(TRAIN_LR)
MASK_NAME = MODEL_NAME + "_mask_r"+str(RATIO)
GPU = 1
MODEL_TYPE = "wideresnet_10_10"

# create init model
cmd_init = "python ticket.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --create_init"
# create last model used for getting pruning mask for different pruning ratio
cmd0 = "python ticket.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --warmup --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
# get the mask from the last model
cmd1 = "python ratio.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --warmup --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
# enhance with mask and original initial weights
cmd2 = "python enhance.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --enhance_learning_rate 0.1 --model_name init_enhance_m"+str(TRAIN_LR)+"_0.01warmup0.1 --init --mask_name " + MASK_NAME
# enhance with mask and random initial weights
cmd3 = "python enhance.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --enhance_learning_rate 0.1 --model_name seed9_enhance_m"+str(TRAIN_LR)+"_0.01warmup0.1 --mask_name " + MASK_NAME

import sys
import os
os.system(cmd_init)
os.system(cmd0)
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
