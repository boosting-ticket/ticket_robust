"""
This repo is based on https://github.com/wanglouis49/pytorch-weights_pruning code.
"""
import os 
import matplotlib.pyplot as plt
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

ratio = [0.005, 0.01, 0.05, 0.1]
true_ratio = []

plot_lines = [-1]

expand = True
inits = []
res_inits = []
epochs = [20, 30, 40, 50, 60, 70, 80, 90, 100]

for r in ratio:
	init_test_accs = []
	res_init_test_accs = []
	for epoch in epochs:
		init_acc, init_test_acc = read_acc(str(r)+"/init_enhance"+str(epoch)+"_m"+str(r)+"_warmup0.1.log", ACC_SIG)
		init_test_accs.append(init_test_acc)
		init_acc, init_test_acc = read_acc("r"+str(r)+"/init_enhance"+str(epoch)+"_m"+str(r)+"_warmup0.1.log", ACC_SIG)
		res_init_test_accs.append(init_test_acc)
	inits.append(init_test_accs)
	res_inits.append(res_init_test_accs)

figure_num = 0

dot = ["v-", "o-", "^-", ".-"]
figure_num = 0
fig, axes = plt.subplots(1,2, figsize=(10,4))
for i in range(2):
	for r in range(len(ratio)):
		#print(inits[r])
		if i==0:
			if r > 2:
				axes[i].plot(epochs, reverse_smooth(np.array(inits[r])), dot[r], linewidth=2.0, label=str(ratio[r]))
			else:
				axes[i].plot(epochs, np.array(inits[r]), dot[r], linewidth=2.0, label=str(ratio[r]))
		else:
			if r==3:
				axes[i].plot(epochs, reverse_smooth(np.array(res_inits[r])), dot[r], linewidth=2.0, label=str(ratio[r]))
			else:
				axes[i].plot(epochs, np.array(res_inits[r]), dot[r], linewidth=2.0, label=str(ratio[r]))
	axes[i].set_xlabel("Training epochs", fontsize=18)
	axes[i].set_ylabel("Test Accuracy (%)", fontsize=18)
	axes[i].tick_params(axis="x", labelsize=15)
	axes[i].tick_params(axis="y", labelsize=15)
	# axes[1].set_ylim((85, 96))
	# axes[1].set_xlim((50, 100))
	# axes[0].set_ylim((60, 96))
	remove_ticks(axes[i])
	modify_splines(axes[i], lwd=0.75, col='0.8')
	remove_splines(axes[i], ['top','right'])
	axes[i].patch.set_facecolor('0.93')
	axes[i].grid(True, 'major', color='0.98', linestyle='-', linewidth=1.0)
	axes[i].set_axisbelow(True)  
plt.tight_layout()
plt.legend(bbox_to_anchor=(-1.1, 1.05, 1.94, -.102), loc="lower left",
	mode="expand", borderaxespad=0., fontsize=15, ncol=5)
plt.savefig("learning_rate_acc.pdf", bbox_inches='tight')
plt.show()
figure_num += 1 
