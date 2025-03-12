import torch.autograd as grad
import torch
import torch.nn as nn
import numpy as np

import nn_loss


from nn_layer import *

from nn_loss import LogLoss, MinsqrLoss

import nn_optimizer as opt



from nn_layer import nnLayer
import numpy as np

from nn_loss import *


class nnImpl:

    def __init__(self, in_dim, out_dim):

        self.with_bias = True
        self.layers: np.ndarray[nnLayer] =         [   
                                LinearLayer(in_dim, 4, with_bias=self.with_bias, kernel=Node.dot),
            ReluLayer(),
            LinearLayer(4, out_dim, with_bias=self.with_bias, kernel=Node.dot),
            SoftargmaxLayer()]

    def forward(self, x_: Node) -> Node:
        for layer in self.layers:
            x_ = layer.forward(x_)
        return x_



def test():

    np.seterr(all='raise')

    n = 10000

    from numpy.random import rand

    x_ = np.array([[rand(), rand(), rand()] for _ in range(n)])

    true = np.array([[1, 0, 0] if x[2] < 0.33 else [0, 1, 0]
                    if x[2] < 0.66 else [0, 0, 1] for x in x_])

    net = nnImpl(3, 3)

    indxs = list(range(0, n))
    epochs = 10 ** 3 * 5

    optim = opt.AdamOptim(net, lr=10**-4 * 5)
    loss = LogLoss()

    for epoch in range(0, epochs + 1):
        pred = net.forward(Node(x_))

        loss_ = loss.backward(pred, Node(true))

        optim.descend()
        if epoch % 10 ** 2 == 0:
            print(f'epoch: {epoch} loss value: {loss_ / n}')
            np.random.shuffle(indxs)
            x_ = x_[indxs]
            true = true[indxs]

    res = net.forward(Node(np.array([[0.1, 0.2, 0.4]])))

    best = res.val.argmax(axis=1)
    print(f'Prediction is {res}, best={best}')


if __name__ == '__main__':
    test()
    pass
