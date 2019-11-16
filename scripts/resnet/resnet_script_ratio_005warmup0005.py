#! /bin/bash
#18, 32, 45, 56, 65, 73, 80, 87, 92, 98

RATIO = 80
TRAIN_LR = 0.05

MODEL_NAME = "pruned_lr0.05_warmup0.005"
MASK_NAME = MODEL_NAME + "_mask_r"+str(RATIO)
GPU = 0

cmd_init = "python ticket.py resnet18 pure --gpu " + str(GPU) + " --create_init"
cmd0 = "python ticket.py resnet18 pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
cmd1 = "python ratio.py resnet18 pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
cmd2 = "python enhance.py resnet18 pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --enhance_learning_rate 0.1 --model_name init_enhance_0005m"+str(TRAIN_LR)+"_warmup0.1 --init --mask_name " + MASK_NAME
cmd3 = "python enhance.py resnet18 pure --gpu " + str(GPU) + " --max_pruning_ratio " + str(RATIO) + " --enhance_learning_rate 0.1 --model_name seed9_enhance_0005m"+str(TRAIN_LR)+"_warmup0.1 --mask_name " + MASK_NAME

import sys
import os
#os.system(cmd_init)
#os.system(cmd0)
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
