## Hacky way to return iris dataset since sklearn is throwing a tantrum in my environment

import random
from copy import deepcopy
import torch

lookup = {'"Setosa"':0,'"Versicolor"':1,'"Virginica"':2}

f = open('./iris.csv', 'r')
l = f.readlines()
f.close()

data = deepcopy(l[1:])

random.seed(42)
random.shuffle(data)
t = int(len(data)*0.85)
train = data[:t]
test  = data[t:]

def get_torch(lineList):
    ll = [line.strip().split(',') for line in lineList]
    xs = [l[:-1] for l in ll]
    ys = [lookup[l[-1]] for l in ll]
    xs = [[float(x) for x in l] for l in xs]
    ys = [[y] for y in ys]
    xs = torch.tensor(xs).float()
    ys = torch.tensor(ys).long()
    return xs, ys

x_train, y_train = get_torch(train)
x_test,  y_test  = get_torch(test)


def to_one_hot(inds, length):
    n = inds.size(0)
    out = torch.zeros(n, length).float()
    for i in range(n):
        out[i, inds[i]] = 1.
    return out

y_train = to_one_hot(y_train, 3)
y_test = to_one_hot(y_test, 3)


