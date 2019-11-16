"""
This repo is based on https://github.com/wanglouis49/pytorch-weights_pruning code.
"""
import os 
import matplotlib.pyplot as plt
import numpy as np
import sys

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
files = []
accs = []

num = 3
if len(sys.argv) == 1:
	file_names = ["boosting", "winning", "rand_init"]
	for name in file_names:
		files.append(name)
else:
	for n in range(len(sys.argv)-1):
		files.append(sys.argv[n+1])

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
		if i < acc.shape[0]-10:
			acc_new.append(np.mean(acc[i:(i+3)]))
		else:
			acc_new.append(acc[i-10])
	return np.array(acc_new)

for file_name in files:
	# accs.append(smooth(read_acc(file_name+".log", ACC_SIG)))
	accs.append(read_acc(file_name+".log", ACC_SIG))
x = np.arange(accs[0].shape[0])

figure_num = 0
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i in range(2):
	for n in range(num):
		if n > 1:
			axes[i].plot(x, accs[n], "--", color='C'+str(n%2), linewidth=2.0, label=files[n])
		else:
			axes[i].plot(x, accs[n], color='C'+str(n%2), linewidth=2.0, label=files[n])
	axes[i].set_xlabel("Training epochs", fontsize=18)
	axes[i].set_ylabel("Validation Accuracy (%)", fontsize=18)
	axes[i].tick_params(axis="x", labelsize=15)
	axes[i].tick_params(axis="y", labelsize=15)
	axes[1].set_ylim((90, 95))
	axes[1].set_xlim((50, 100))
	axes[0].set_ylim((60, 96))

	remove_ticks(axes[i])
	modify_splines(axes[i], lwd=0.75, col='0.8')
	remove_splines(axes[i], ['top','right'])
	axes[i].patch.set_facecolor('0.93')
	axes[i].grid(True, 'major', color='0.98', linestyle='-', linewidth=1.0)
	axes[i].set_axisbelow(True)  
plt.tight_layout()
# plt.legend(bbox_to_anchor=(-0.9, 1.05, 1.54, -.102), loc="lower left",
# 	mode="expand", borderaxespad=0., fontsize=15, ncol=3)
plt.savefig("resnet.pdf", bbox_inches='tight')
plt.show()