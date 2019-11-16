"""
This repo is based on https://github.com/wanglouis49/pytorch-weights_pruning code.
"""
import os 
import matplotlib.pyplot as plt
import numpy as np
import sys
import pdb

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

VALID_SIG = "Valid"
ACC_SIG = "Test accuracy: "

expand = True
num = 4
files = ["init_enhance100_m0.01_warmup0.1.log", "single_init_enhance_m0.01_warmup0.1.log",
		 "seed9_enhance100_m0.01_warmup0.1.log", "single_seed9_enhance_m0.01_warmup0.1.log",
		 "init_enhance_005m0.05_warmup0.1.log", "single_init_enhance_m0.05_warmup0.1.log",
		 "seed9_enhance_005m0.05_warmup0.1.log", "single_seed9_enhance_m0.05_warmup0.1.log"
		 ]
accs = []

# for n in range(num):
# 	files.append(sys.argv[n+1]+".log")


def read_acc(file_name, sig, length=5):
	is_valid = False
	file = open(file_name, "r")
	acc = [0]
	sig_len = len(sig)
	for l in file:
		if VALID_SIG in l:
			is_valid = True
		if (sig in l) and is_valid:
			acc.append(float(l[sig_len:sig_len+length]))
			is_valid = False
	file.close()
	return np.array(acc)

def smooth(acc):
	acc_new = []
	for i in range(acc.shape[0]):
		if i < acc.shape[0]-3:
			acc_new.append(np.mean(acc[i:(i+3)]))
		else:
			acc_new.append(acc[i])
	return np.array(acc_new)

for file_name in files:
	accs.append(smooth(read_acc(file_name, ACC_SIG)))

x = np.arange(accs[0].shape[0])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

labels = ["iterative", "one shot", "iterative_rand", "one shot_rand"]
for i in range(2):
	for n in range(num):
		if n > 1:
			axes[i].plot(x, accs[i*4+n], "--", color='C'+str(n%2), linewidth=2.0, label=labels[n])
		else:
			axes[i].plot(x, accs[i*4+n], color='C'+str(n%2), linewidth=2.0, label=labels[n])
	axes[i].set_xlabel("Training epochs", fontsize=18)
	axes[i].set_ylabel("Validation Accuracy (%)", fontsize=18)
	axes[i].tick_params(axis="x", labelsize=15)
	axes[i].tick_params(axis="y", labelsize=15)
	# axes[1].set_ylim((90, 93))
	# axes[1].set_xlim((50, 100))
	axes[i].set_ylim((60, 96))

	remove_ticks(axes[i])
	modify_splines(axes[i], lwd=0.75, col='0.8')
	remove_splines(axes[i], ['top','right'])
	axes[i].patch.set_facecolor('0.93')
	axes[i].grid(True, 'major', color='0.98', linestyle='-', linewidth=1.0)
	axes[i].set_axisbelow(True)  
plt.tight_layout()
plt.legend(bbox_to_anchor=(-1.15, 1.05, 2.04, -.102), loc="lower left",
	mode="expand", borderaxespad=0., fontsize=15, ncol=4)
plt.savefig("iterative.pdf", bbox_inches='tight')
plt.show()
