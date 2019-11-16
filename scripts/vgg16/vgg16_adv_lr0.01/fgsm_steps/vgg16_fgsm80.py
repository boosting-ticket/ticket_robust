#! /bin/bash
#18, 32, 45, 56, 65, 73, 80, 87, 92, 98

RATIO = 80
TRAIN_LR = 0.01

MODEL_NAME = "pruned_lr" + str(TRAIN_LR)
MASK_NAME = MODEL_NAME+"_nn" + "_mask_r"+str(RATIO)
GPU = 0
FINETUNE_METHOD = "fgsm"
ENHANCE_METHOD = "madry"
TRAIN_EPOCH = 80

# create init model
cmd_init = "python ticket.py vgg16 pure --gpu " + str(GPU) + " --create_init"
# create last model used for getting pruning mask for different pruning ratio
cmd0 = "python ticket.py vgg16 pure --norm --train_epochs "+str(TRAIN_EPOCH)+" --gpu " + str(GPU) + " --finetune_method "+FINETUNE_METHOD+" --max_pruning_ratio " + str(RATIO) + " --warmup --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
# get the mask from the last model
cmd1 = "python ratio.py vgg16 pure --norm --train_epochs "+str(TRAIN_EPOCH)+" --gpu " + str(GPU) + " --finetune_method "+FINETUNE_METHOD+" --max_pruning_ratio " + str(RATIO) + " --warmup --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
# enhance with mask and original initial weights
cmd2 = "python enhance.py vgg16 pure --norm --train_epochs "+str(TRAIN_EPOCH)+" --enhance_epochs 100 --gpu " + str(GPU) + " --finetune_method "+FINETUNE_METHOD+" --enhance_method "+ENHANCE_METHOD+" --max_pruning_ratio " + str(RATIO) + " --enhance_learning_rate 0.1 --model_name init_enhance_"+ENHANCE_METHOD+"_m"+str(TRAIN_LR)+"_warmup0.1 --init --mask_name " + MASK_NAME
# enhance with mask and random initial weights
cmd3 = "python enhance.py vgg16 pure --norm --train_epochs "+str(TRAIN_EPOCH)+" --enhance_epochs 100 --gpu " + str(GPU) + " --finetune_method "+FINETUNE_METHOD+" --enhance_method "+ENHANCE_METHOD+" --max_pruning_ratio " + str(RATIO) + " --enhance_learning_rate 0.1 --model_name seed9_enhance_"+ENHANCE_METHOD+"_m"+str(TRAIN_LR)+"_warmup0.1 --mask_name " + MASK_NAME

import sys
import os
#os.system(cmd_init)
os.system(cmd0)
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
