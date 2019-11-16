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
epochs = [20, 30, 40, 50, 60, 70, 80, 90, 100]

for r in ratio:
	init_test_accs = []
	for epoch in epochs:
		init_acc, init_test_acc = read_acc(str(r)+"/init_enhance"+str(epoch)+"_m"+str(r)+"_warmup0.1.log", ACC_SIG)
		init_test_accs.append(init_test_acc)
	inits.append(init_test_accs)


figure_num = 0

dot = ["v-", "o-", "^-", ".-"]
for expand in [False]:
	plt.figure(figure_num, figsize=(5,4))
	pdb.set_trace()
	for r in range(len(ratio)):
		#print(inits[r])
		if r < 2:
			plt.plot(epochs, reverse_smooth(np.array(inits[r])), dot[r], linewidth=2.0, label=str(ratio[r]))
		else:
			plt.plot(epochs, np.array(inits[r]), dot[r], linewidth=2.0, label=str(ratio[r]))
	plt.xlabel("Training epochs", fontsize=18)
	plt.ylabel("Test Accuracy (%)", fontsize=18)
	
	plt.legend(loc="best", fontsize=12)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	# if expand:
		# plt.ylim((85, 95))
		# plt.xlim((50, 100))
	# else:
	#plt.ylim((40, 85))
	plt.tight_layout()

	if expand:
		plt.savefig("learning_rate_expand.pdf")
	else:
		plt.savefig("learning_rate.pdf")
	plt.show()
	figure_num += 1 
