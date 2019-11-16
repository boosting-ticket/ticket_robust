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

widths = [1, 2, 5, 10]

expand = True
inits = []
seed9s = []


figure_num = 0
for expand in [False, True]:
	plt.figure(figure_num, figsize=(5,4))

	for w in widths:
		init_acc, init_test_acc = read_acc("wideresnet_34_"+str(w)+"/init_enhance_m0.05_warmup0.1.log", ACC_SIG)
		seed9_acc, seed9_test_acc = read_acc("wideresnet_34_"+str(w)+"/seed9_enhance_m0.05_warmup0.1.log", ACC_SIG)
		plt.plot(np.arange(len(init_acc)), init_acc, linewidth=2.0, label="34_"+str(w))
		#plt.plot(np.arange(len(init_acc)), init_acc, linewidth=2.0, label=str(d)+"rand")

	plt.xlabel("Training epochs", fontsize=18)
	plt.ylabel("Validation Accuracy (%)", fontsize=18)
	
	plt.legend(loc="best", fontsize=12)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	if expand:
		plt.ylim((85, 96))
		plt.xlim((50, 100))
	else:
		plt.ylim((60, 96))
	plt.tight_layout()

	if expand:
		plt.savefig("prune_width_acc_expand.pdf")
	else:
		plt.savefig("prune_width_acc.pdf")
	plt.show()
	figure_num += 1 