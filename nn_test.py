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

    n = 1000

    from numpy.random import rand

    x_ = np.array([[rand(), rand(), rand()] for _ in range(n)])

    true = np.array([[1, 0, 0] if x[2] < 0.33 else [0, 1, 0]
                    if x[2] < 0.66 else [0, 0, 1] for x in x_])

    net = nnImpl(3, 3, step=10**-4)

    indxs = list(range(0, n))
    epochs = 10 ** 3 * 4

    loss = LogLoss()

    for epoch in range(1, epochs + 1):
        pred = net.forward(Node(x_))

        loss_ = loss.backward(pred, Node(true))

        net.descent()
        if epoch % 10 ** 2 == 0:
            print(f'epoch: {epoch} loss value: {loss_ / n}')
            np.random.shuffle(indxs)
            x_ = x_[indxs]
            true = true[indxs]


    res = net.forward(Node(np.array([[0.1, 0.2, 0.3]])))

    best = res.val.argmax(axis=1)
    print(f'Prediction is {res}, best={best}')


if __name__ == '__main__':
    test()
    pass
