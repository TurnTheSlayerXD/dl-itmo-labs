import torch.autograd as grad
import torch
import torch.nn as nn
import numpy as np

import nn_loss


from nn_layer import *
from nn_impl import nnImpl

from nn_loss import LogLoss, MinsqrLoss


def test():

    np.seterr(all='raise')

    n = 100

    from numpy.random import rand

    x_ = np.array([[rand(), rand(), rand()] for _ in range(n)])

    true = np.array([x * 5 for x in x_])


    net = nnImpl(3, 3, step=10**-12)

    indxs = list(range(0, n))
    epochs = 10 ** 5

    loss = MinsqrLoss()

    for epoch in range(1, epochs + 1):
        pred = net.forward(Node(x_))

        loss_ = loss.backward(pred, Node(true))

        net.descent()
        if epoch % 10 ** 2 == 0:
            print(f'epoch: {epoch} loss value: {loss_ / n}')
            np.random.shuffle(indxs)
            x_ = x_[indxs]
            true = true[indxs]

    # ev = [ [n] for n in range(2, l, 2)]
    # y_ev = [[1, 0]] * len(ev)

    # odd = [ [n] for n in range(1, l, 2)]
    # y_odd = [[0, 1]] * len(odd)

    # x_ = ev + odd
    # y_ = y_ev + y_odd

   
    res = net.forward(Node(np.array([[0.1, 0.2, 0.3]])))

    best = res.val.argmax(axis=1)
    print(f'Prediction is {res}, best={best}')



if __name__ == '__main__':
    test()
    pass
