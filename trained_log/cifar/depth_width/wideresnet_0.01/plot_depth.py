"""
This repo is based on https://github.com/wanglouis49/pytorch-weights_pruning code.
"""
import os 
import matplotlib.pyplot as plt
import numpy as np
import sys


VALID_SIG = "Valid"
ACC_SIG = "Test accuracy: "
TEST_SIG = "Test on test set"

def remove_ticks(ax):
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
def remove_splines(ax, spl):
    for s in spl:
        ax.spines[s].set_visible(False)  

def modify_splines(ax, lwd, col):    
    for s in ['bottom', 'left','top','right']:
        ax.spines[s].set_linewidth(lwd)
        ax.spines[s].set_color(col)  

def read_acc(file_name, sig, length=5):
	is_valid = False
	is_test = False
	file = open(file_name, "r")
	acc = [0]
	sig_len = len(sig)
	test_acc = -1
	for l in file:
		if VALID_SIG in l:
			is_valid = True
		if TEST_SIG in l:
			is_test = True
		if (sig in l) and is_valid:
			acc.append(float(l[sig_len:sig_len+length]))
			is_valid = False
		if (sig in l) and is_test:
			test_acc = float(l[sig_len:sig_len+length])
			is_test = False
	file.close()
	#acc.append(test_acc)
	return np.array(acc), test_acc



def smooth(acc):
	acc_new = []
	for i in range(acc.shape[0]):
		if i < acc.shape[0]-3:
			acc_new.append(np.mean(acc[i:(i+3)]))
		else:
			acc_new.append(acc[i])
	return np.array(acc_new)

depths = [10, 16, 22, 28, 34]

expand = True
inits = []
seed9s = []


figure_num = 0
fig, axes = plt.subplots(1,2, figsize=(10,4))
for i in range(2):
	for d in depths:
		init_acc, init_test_acc = read_acc("wideresnet_"+str(d)+"_10/init_enhance_m0.05_0.01.log", ACC_SIG)
		#seed9_acc, seed9_test_acc = read_acc("wideresnet_"+str(d)+"_10/seed9_enhance_m0.05_warmup0.1.log", ACC_SIG)
		axes[i].plot(np.arange(len(init_acc)), init_acc, linewidth=2.0, label=str(d)+"-10")
		#plt.plot(np.arange(len(init_acc)), init_acc, linewidth=2.0, label=str(d)+"rand")

	axes[i].set_xlabel("Training epochs", fontsize=18)
	axes[i].set_ylabel("Validation Accuracy (%)", fontsize=18)
	axes[i].tick_params(axis="x", labelsize=15)
	axes[i].tick_params(axis="y", labelsize=15)
	axes[1].set_ylim((85, 96))
	axes[1].set_xlim((50, 100))
	axes[0].set_ylim((60, 96))
	remove_ticks(axes[i])
	modify_splines(axes[i], lwd=0.75, col='0.8')
	remove_splines(axes[i], ['top','right'])
	axes[i].patch.set_facecolor('0.93')
	axes[i].grid(True, 'major', color='0.98', linestyle='-', linewidth=1.0)
	axes[i].set_axisbelow(True)  
plt.tight_layout()
plt.legend(bbox_to_anchor=(-1.1, 1.05, 1.94, -.102), loc="lower left",
	mode="expand", borderaxespad=0., fontsize=15, ncol=5)
plt.savefig("prune_depth_acc.pdf", bbox_inches='tight')
plt.show()
figure_num += 1 