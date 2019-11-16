"""
This repo is based on https://github.com/wanglouis49/pytorch-weights_pruning code.
"""
import os 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import pdb

VALID_SIG = "Valid"
ACC_SIG = "Clean accuracy: "
ADV_SIG = "Adversarial accuracy: "
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
	adv_acc = [0]
	test_acc = -1
	test_adv_acc = -1
	for l in file:
		if VALID_SIG in l:
			is_valid = True
			continue
		if TEST_SIG in l:
			is_test = True
			continue
		if (ACC_SIG in l) and is_test:
			sig_len = len(ACC_SIG)
			test_acc = float(l[sig_len:sig_len+length])
			continue
		if (ADV_SIG in l) and is_test:
			sig_len = len(ADV_SIG)
			test_adv_acc = float(l[sig_len:sig_len+length])
			is_test = False
			continue 
		if (ACC_SIG in l) and is_valid:
			sig_len = len(ACC_SIG)
			acc.append(float(l[sig_len:sig_len+length]))
			continue
		if (ADV_SIG in l) and is_valid:
			sig_len = len(ADV_SIG)
			adv_acc.append(float(l[sig_len:sig_len+length]))
			is_valid = False
			continue
		
	file.close()
	#acc.append(test_acc)
	return np.array(acc), np.array(adv_acc), test_acc



def reverse_smooth(acc):
	acc_new = []
	#print(acc)
	for i in range(acc.shape[0]):
		if i >= 1:
			acc_new.append(np.mean(acc[(i-1):(i+1)]))
		if i==0:
			acc_new.append(acc[i])
	return np.array(acc_new)


def smooth(acc):
	acc_new = []
	for i in range(acc.shape[0]):
		if i < acc.shape[0]-3:
			acc_new.append(np.mean(acc[(i):(i+3)]))
		else:
			acc_new.append(acc[i])
	return np.array(acc_new)

names = ["nat", "fgsm", "pgd"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
lines = []
for i in range(2):
	init_accs = []
	init_adv_accs = []
	for j, name in enumerate(names):
		init_acc, init_adv_acc, init_test_acc = read_acc(name+"/init_enhance_madry_m0.01_warmup0.1_nn.log", ACC_SIG)
		seed_acc, seed_adv_acc, seed_test_acc = read_acc("nat/seed9_enhance_madry_m0.01_warmup0.1_nn.log", ACC_SIG)
		axes[0].plot(range(init_acc.shape[0]), np.array(init_acc), color="C"+str(j), linewidth=2.0)
		axes[1].plot(range(init_adv_acc.shape[0]), np.array(init_adv_acc), color="C"+str(j), linewidth=2.0)
		if i==0:
			lines.append(mlines.Line2D([], [], color="C"+str(j), label=name))
			
	axes[0].plot(range(seed_acc.shape[0]), np.array(seed_acc), "--", color="C"+str(3), linewidth=2.0)
	axes[1].plot(range(seed_adv_acc.shape[0]), np.array(seed_adv_acc), "--", color="C"+str(3), linewidth=2.0)
	if i==0:
		lines.append(mlines.Line2D([], [], ls="--", color="C"+str(3), label="rand"))
	axes[0].set_ylabel("Clean Accuracy (%)",fontsize=16)
	axes[1].set_ylabel("Robust Accuracy (%)",fontsize=16)
	axes[i].set_xlabel("Training epochs",fontsize=16)
	axes[i].tick_params(axis="x", labelsize=15)
	axes[i].tick_params(axis="y", labelsize=15)
	axes[0].set_ylim((35, 80))
	axes[1].set_ylim((20, 50))
	remove_ticks(axes[i])
	modify_splines(axes[i], lwd=0.75, col='0.8')
	remove_splines(axes[i], ['top','right'])
	axes[i].patch.set_facecolor('0.93')
	axes[i].grid(True, 'major', color='0.98', linestyle='-', linewidth=1.0)
	axes[i].set_axisbelow(True)  


plt.tight_layout()
plt.legend(handles=lines, bbox_to_anchor=(-0.9, 1.05, 1.54, -.102), loc="lower left",
	mode="expand", borderaxespad=0., fontsize=15, ncol=4)
# plt.legend(handles=lines, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=15)
plt.savefig("nat_fgsm_madry_large.pdf", bbox_inches='tight')
plt.show()

