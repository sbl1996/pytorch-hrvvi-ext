import re
import matplotlib.pyplot as plt
import numpy as np

from horch.functools import lmap
from horch.io import read_lines

epoch_p = re.compile(r"""Epoch \d+, lr (\d\.\d{6})""")
def get_lr(s):
    return float(epoch_p.search(s).group(1))

train_p = re.compile(r""".* train - loss: (\d\.\d{4}), acc: (\d\.\d{4})""")
def extract_train(s):
    m = train_p.search(s)
    return float(m.group(1)), float(m.group(2))

valid_p = re.compile(r""".* valid - loss: (\d\.\d{4}), acc: (\d\.\d{4})""")
def extract_valid(s):
    m = valid_p.search(s)
    return float(m.group(1)), float(m.group(2))

def extract(fp):
    lines = read_lines(fp)
    epoch_lines = []
    train_lines = []
    valid_lines = []

    for l in lines:
        if 'Epoch' in l:
            epoch_lines.append(l)
        elif 'train' in l:
            train_lines.append(l)
        elif 'valid' in l:
            valid_lines.append(l)


    lrs = lmap(get_lr, epoch_lines)

    train_losses, train_accs = zip(*map(extract_train, train_lines))
    valid_losses, valid_accs = zip(*map(extract_valid, valid_lines))

    lrs, train_losses, train_accs, valid_losses, valid_accs = [
        np.array(x) for x in [lrs, train_losses, train_accs, valid_losses, valid_accs]]
    return lrs, train_losses, train_accs, valid_losses, valid_accs


lrs, train_losses, train_accs, valid_losses, valid_accs = extract("/Users/hrvvi/Code/Library/experiments/CIFAR10/5.log")
errs = (1 - valid_accs) * 100
plt.plot(errs)
plt.ylim([-0.9, 20])