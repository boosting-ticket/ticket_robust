#! /bin/bash
#18, 32, 45, 56, 65, 73, 80, 87, 92, 98

RATIO = 80
TRAIN_LR = 0.05
# Need to test whether it's the best or 0.01
ENHANCE_LR = 0.05

MODEL_NAME = "pruned_trades6_lr" + str(TRAIN_LR)
MASK_NAME = MODEL_NAME+"_nn" + "_mask_r"+str(RATIO)

ENHANCE_METHOD = "trades"
FINETUNE_METHOD = "fgsm"

GPU = "0,1"

MODEL_TYPE = "wideresnet"
ENHANCE_EPOCH = 100

# create init model
cmd_init = "python ticket.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --create_init"
# create last model used for getting pruning mask for different pruning ratio
cmd0 = "python ticket.py "+MODEL_TYPE+" pure --gpu " + str(GPU) + " --norm --finetune_method "+FINETUNE_METHOD+" --max_pruning_ratio " + str(RATIO) + " --warmup --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
# get the mask from the last model
cmd1 = "python ratio.py "+MODEL_TYPE+" pure --norm --gpu " + str(GPU) + " --finetune_method "+FINETUNE_METHOD+\
			" --max_pruning_ratio " + str(RATIO) + " --warmup --learning_rate " + str(TRAIN_LR) + " --model_name " + MODEL_NAME
# enhance with mask and original initial weights
cmd2 = "python enhance.py "+MODEL_TYPE+" pure --norm --gpu " + str(GPU) + " --finetune_method "+FINETUNE_METHOD+\
		" --enhance_learning_rate "+str(ENHANCE_LR)+" --enhance_epochs "+str(ENHANCE_EPOCH)+" --enhance_method "+ENHANCE_METHOD+" --max_pruning_ratio "+\
		str(RATIO) + " --model_name init_enhance"+str(ENHANCE_EPOCH)+"_"+ENHANCE_METHOD+"_m"+str(TRAIN_LR)+"_nn_warmup0.05 --init --mask_name " + MASK_NAME
# enhance with mask and random initial weights
cmd3 = "python enhance.py "+MODEL_TYPE+" pure --norm --gpu " + str(GPU) + " --finetune_method "+FINETUNE_METHOD+\
		" --enhance_learning_rate "+str(ENHANCE_LR)+" --enhance_epochs "+str(ENHANCE_EPOCH)+" --enhance_method "+ENHANCE_METHOD+" --max_pruning_ratio "+\
		str(RATIO) + " --model_name seed9_enhance"+str(ENHANCE_EPOCH)+"_"+ENHANCE_METHOD+"_m"+str(TRAIN_LR)+"_nn_warmup0.05 --mask_name " + MASK_NAME


import sys
import os
#os.system(cmd_init)
os.system(cmd0)
os.system(cmd1)
#os.system(cmd2)
#os.system(cmd3)