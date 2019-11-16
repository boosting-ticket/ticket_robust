## Boosting Ticket: Towards Practical Pruning for Adversarial Training with Lottery Ticket Hypothesis

### How to reproduce the results in paper
We have summarized all of the running command in file `scripts`.
For example, to run the regular pruning to achieve the boosting ticket and then retraining on VGG-16, one can easily run the command:

```python
python vgg16_script_ratio80.py
```

All of the random seeds are fixed such that one can easily get the same results as we reported in paper.

### Results in paper
All of the results can be found in file `trained_log`. It includes all the training logs ran by the scripts in file `scripts` and the python scripts for constructing the plots in paper.

