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

ratio = [0.01, 0.1]
true_ratio = []

plot_lines = [-1]

expand = True
inits = []
epochs = [20, 40, 60, 80, 100]

for r in ratio:
	init_accs = []
	for epoch in epochs:
		init_acc, init_test_acc = read_acc(str(r)+"/init_enhance"+str(epoch)+"_m"+str(r)+"_warmup0.1.log", ACC_SIG)
		if len(init_acc) < 100:
			init_acc = np.concatenate((init_acc , np.array([None]*(101-len(init_acc)))))
		init_accs.append(init_acc)
	inits.append(init_accs)


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
			for r in range(len(ratio)):
				if j==0 and i<2:
					axes[j,i].plot(range(101), np.array(inits[r][e]), line[r], color="C"+str(e), linewidth=2.0)
					if i==0:
						lines.append(mlines.Line2D([], [], ls=line[r], color="C"+str(e), label=str(label[r])+str(epochs[e])))
				else:
					axes[j,i].plot(range(101), np.array(inits[r][-1]), line[r], color="C"+str(4), linewidth=2.0)
					axes[j,i].plot(range(101), np.array(inits[r][j*2+i-2]), line[r], color="C"+str(j*2+i-2), linewidth=2.0)		
		axes[j,i].set_ylabel("Validation Accuracy (%)",fontsize=16)
		axes[j,i].set_xlabel("Training epochs",fontsize=16)
		axes[j,i].tick_params(axis="x", labelsize=15)
		axes[j,i].tick_params(axis="y", labelsize=15)
		axes[j,i].set_ylim((88.3, 92.5))
		axes[0,0].set_ylim((65, 95))
		remove_ticks(axes[j,i])
		modify_splines(axes[j,i], lwd=0.75, col='0.8')
		remove_splines(axes[j,i], ['top','right'])
		axes[j,i].patch.set_facecolor('0.93')
		axes[j,i].grid(True, 'major', color='0.98', linestyle='-', linewidth=1.0)
		axes[j,i].set_axisbelow(True)  

plt.tight_layout()
plt.legend(handles=lines, bbox_to_anchor=(-1.1, 3.85, 2.04, -.102), loc="lower left",
	mode="expand", borderaxespad=0., fontsize=15, ncol=5)
plt.savefig("convergence_acc_large.pdf", bbox_inches='tight')
plt.show()

