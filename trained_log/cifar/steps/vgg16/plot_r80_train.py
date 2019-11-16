"""
This repo is based on https://github.com/wanglouis49/pytorch-weights_pruning code.
"""
import os 
import matplotlib.pyplot as plt
import numpy as np
import sys


accs = np.zeros((4, 5))

accs[0] = np.array([86.28, 89.4, 90.30, 90.44, 91.64])
accs[1] = np.array([86.47, 89.13, 89.72, 90.35, 91.98])
accs[2] = np.array([90.53, 91.60, 91.78, 91.76, 92.1])
accs[3] = np.array([89.88, 90.31, 91.02, 90.96, 91.32])

def smooth(acc):
	acc_new = []
	for i in range(acc.shape[0]):
		if i < acc.shape[0]-3:
			acc_new.append(np.mean(acc[i:(i+3)]))
		else:
			acc_new.append(acc[i])
	return np.array(acc_new)

x = np.array([20, 40, 60, 80, 100])

plt.figure(0, figsize=(5,4))
plt.plot(x, accs[0], color='C1', linewidth=2.0, label="lr=0.1")
plt.plot(x, accs[1], color='C2', linewidth=2.0, label="lr=0.05")
plt.plot(x, accs[2], color='C3', linewidth=2.0, label="lr=0.01")
plt.plot(x, accs[3], color='C4', linewidth=2.0, label="lr=0.005")

plt.xlabel("Training epochs", fontsize=18)
plt.ylabel("Test Accuracy (%)", fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# if expand:
# 	plt.ylim((90, 95))
# 	plt.xlim((50, 100))
# else:
# 	plt.ylim((60, 95))
# plt.tight_layout()

plt.savefig("steps_acc.pdf")
plt.show()
