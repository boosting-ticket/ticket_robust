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
TEST_SIG = "Test on test set"


INIT_NAME = "init_enhance_m0.01_warmup0.1"
SEED9_NAME = "seed9_enhance_m0.01_warmup0.1"

RESNET_INIT_NAME = "init_enhance_m0.05_warmup0.1"
RESNET_SEED9_NAME = "seed9_enhance_m0.05_warmup0.1"

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
	acc.append(test_acc)
	return np.array(acc), test_acc



def smooth(acc):
	acc_new = []
	for i in range(acc.shape[0]):
		if i < acc.shape[0]-3:
			acc_new.append(np.mean(acc[i:(i+3)]))
		else:
			acc_new.append(acc[i])
	return np.array(acc_new)

ratio = [0, 18, 32, 45, 56, 65, 73, 80, 87, 92, 98]
true_ratio = []

plot_lines = [1, 5]

expand = True
inits = []
seed9s = []

res_inits = []
res_seed9s = []
for r in ratio:
	if os.path.exists(str(r)+"/"):
		inits.append(str(r)+"/"+INIT_NAME)
		seed9s.append(str(r)+"/"+SEED9_NAME)
		res_inits.append(str(r)+"/"+RESNET_INIT_NAME)
		res_seed9s.append(str(r)+"/"+RESNET_SEED9_NAME)
		true_ratio.append(r)

init_accs = []
seed9_accs = []
res_init_accs = []
res_seed9_accs = []
init_test_accs = []
seed9_test_accs = []
res_init_test_accs = []
res_seed9_test_accs = []

assert len(inits) == len(seed9s), "number of inits and seed9s does not match"
for init_name, seed9_name in zip(inits, seed9s):

	init_acc, init_test_acc = read_acc(init_name+".log", ACC_SIG)
	init_accs.append(init_acc)
	init_test_accs.append(init_test_acc)

	seed9_acc, seed9_test_acc = read_acc(seed9_name+".log", ACC_SIG)
	seed9_accs.append(seed9_acc)
	seed9_test_accs.append(seed9_test_acc)

for init_name, seed9_name in zip(res_inits, res_seed9s):

	init_acc, init_test_acc = read_acc(init_name+".log", ACC_SIG)
	if len(init_acc)<102:
		init_acc = np.concatenate((init_acc, np.array([None]*(102-len(init_acc)))))
	res_init_accs.append(init_acc)
	res_init_test_accs.append(init_test_acc)

	seed9_acc, seed9_test_acc = read_acc(seed9_name+".log", ACC_SIG)
	if len(seed9_acc)<102:
		seed9_acc = np.concatenate((seed9_acc, np.array([None]*(102-len(seed9_acc)))))
	res_seed9_accs.append(seed9_acc)
	res_seed9_test_accs.append(seed9_test_acc)
init_accs = np.array(init_accs)
seed9_accs = np.array(seed9_accs)
res_init_accs = np.array(res_init_accs)
res_seed9_accs = np.array(res_seed9_accs)
init_test_accs = np.array(init_test_accs)
seed9_test_accs = np.array(seed9_test_accs)
res_init_test_accs = np.array(res_init_test_accs)
res_seed9_test_accs = np.array(res_seed9_test_accs)
print(true_ratio, init_accs.shape)

figure_num = 0

fig, axes = plt.subplots(1,2, figsize=(10,4))
for i in range(2):
	for e, epoch in enumerate(plot_lines):
		if epoch == -1:
			epoch_name = 100
		else:
			epoch_name = epoch
		if i==0:
			axes[i].plot(true_ratio, smooth(init_accs[:, epoch]), color="C"+str(e+1), linewidth=2.0, label="epoch"+str(epoch_name))
			axes[i].plot(true_ratio, smooth(seed9_accs[:, epoch]), "--", color="C"+str(e+1), linewidth=2.0, label="epoch"+str(epoch_name)+"_rand")
		else:
			axes[i].plot(true_ratio, smooth(res_init_accs[:, epoch]), color="C"+str(e+1), linewidth=2.0, label="epoch"+str(epoch_name))
			axes[i].plot(true_ratio, smooth(res_seed9_accs[:, epoch]), "--", color="C"+str(e+1), linewidth=2.0, label="epoch"+str(epoch_name)+"_rand")
	axes[i].set_xlabel("Pruning Ratio", fontsize=18)
	axes[i].set_ylabel("Validation Accuracy (%)", fontsize=18)
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
plt.legend(bbox_to_anchor=(-1.15, 1.05, 2.04, -.102), loc="lower left",
	mode="expand", borderaxespad=0., fontsize=15, ncol=4)
plt.savefig("ratio_acc.pdf", bbox_inches='tight')
plt.show()
figure_num += 1 