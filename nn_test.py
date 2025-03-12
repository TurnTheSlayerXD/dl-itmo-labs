import torch.autograd as grad
import torch
import torch.nn as nn
import numpy as np

import nn_loss


from nn_layer import *
from nn_impl import nnImpl

from nn_loss import LogLoss, MinsqrLoss

import nn_optimizer as opt


def test():
<<<<<<< HEAD
    
    np.seterr(all='warn')
    
    layers = [ 
              SigmoidLayer(3, 3, kernel=nn_kernel.GausKernel(q=1)),
                LinearLayer(3, 5, kernel=nn_kernel.GausKernel(q=1)),
                SigmoidLayer(5, 3, kernel=nn_kernel.GausKernel(q=1)),
                LinearLayer(3, 5, kernel=nn_kernel.GausKernel(q=1)),
                SoftArgMaxCrossEntropy(5, 3, kernel=nn_kernel.GausKernel(q=1))]
    
    layers = [ 
              SigmoidLayer(3, 1, kernel=nn_kernel.LinearKernel()),
                LinearLayer(1, 1, kernel=nn_kernel.LinearKernel()),
              SigmoidLayer(1, 2, kernel=nn_kernel.LinearKernel()),
                SoftArgMaxCrossEntropy(2, 3, kernel=nn_kernel.LinearKernel())]
    
    net = nnImpl(layers, epochs=10 * 10 ** 4, speed=10 ** -5)
    
    l = 10000
    # ev = [ [n] for n in range(2, l, 2)]
    # y_ev = [[1, 0]] * len(ev)
    
    # odd = [ [n] for n in range(1, l, 2)]
    # y_odd = [[0, 1]] * len(odd)
    
    # x_ = ev + odd
    # y_ = y_ev + y_odd
    
    from numpy.random import randint
=======

    np.seterr(all='raise')

    n = 10000

>>>>>>> 911aa5a40cbcf1a3d761eab47d80bb8a45dfbfd3
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
