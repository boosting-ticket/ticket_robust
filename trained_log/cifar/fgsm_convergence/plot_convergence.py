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

true_ratio = []

plot_lines = [-1]

expand = True
inits = []
epochs = [20, 40, 60, 80, 100]


for epoch in epochs:
	init_acc, init_adv_acc, init_test_acc = read_acc("init_enhance"+str(epoch)+"_madry_m0.01_nn_warmup0.1_nn.log", ACC_SIG)
	if len(init_acc) < 100:
		init_acc = np.concatenate((init_acc , np.array([None]*(101-len(init_adv_acc)))))
	inits.append(init_acc)

init_acc_b, init_adv_acc_b, init_test_acc_b=read_acc("baseline_pgd.log", ACC_SIG)

figure_num = 0

dot = ["v-", "o-", "^-", ".-"]
line = ["-", "--"]
label = ["boost", "win"]

fig, axes = plt.subplots(3,2,figsize=(10, 10))
lines = []
# pdb.set_trace()
for j in range(3):
	for i in range(2):
		for e in range(len(epochs)):
			#print(inits[r])
			if j==0 and i==0:
				axes[j,i].plot(range(101), np.array(inits[e]), color="C"+str(e), linewidth=2.0)	
				lines.append(mlines.Line2D([], [], color="C"+str(e), label=str(epochs[e])))
			else:
				axes[j,i].plot(range(101), np.array(inits[j*2+i-1]), color="C"+str(j*2+i-1), linewidth=2.0)
		axes[j,i].plot(range(101), np.array(init_acc_b), color="C"+str(6), linewidth=2.0)
		axes[j,i].set_ylabel("Validation Accuracy (%)",fontsize=16)
		axes[j,i].set_xlabel("Training epochs",fontsize=16)
		axes[j,i].tick_params(axis="x", labelsize=15)
		axes[j,i].tick_params(axis="y", labelsize=15)
		axes[j,i].set_ylim((70, 79))
		axes[0,0].set_ylim((35, 80))
		remove_ticks(axes[j,i])
		modify_splines(axes[j,i], lwd=0.75, col='0.8')
		remove_splines(axes[j,i], ['top','right'])
		axes[j,i].patch.set_facecolor('0.93')
		axes[j,i].grid(True, 'major', color='0.98', linestyle='-', linewidth=1.0)
		axes[j,i].set_axisbelow(True)  
lines.append(mlines.Line2D([], [], color="C"+str(6), label="baseline"))	
plt.tight_layout()
plt.legend(handles=lines, bbox_to_anchor=(-1.15, 3.85, 2.04, -.102), loc="lower left",
	mode="expand", borderaxespad=0., fontsize=15, ncol=6)
plt.savefig("convergence_clean_acc_large.pdf", bbox_inches='tight')
plt.show()

