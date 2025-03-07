import numpy as np

import nn_loss

import nn_kernel

from nn_layer import *
from nn_impl import nnImpl


# proof of work
def test():
    
    np.seterr(all='warn')
    
    layers = [ 
              SigmoidLayer(3, 3, kernel=nn_kernel.GausKernel(q=1)),
                LinearLayer(3, 5, kernel=nn_kernel.GausKernel(q=1)),
                SigmoidLayer(5, 3, kernel=nn_kernel.GausKernel(q=1)),
                LinearLayer(3, 5, kernel=nn_kernel.GausKernel(q=1)),
                SoftArgMaxCrossEntropy(5, 3, kernel=nn_kernel.GausKernel(q=1))]
    
    layers = [ 
              SigmoidLayer(3, 3, kernel=nn_kernel.LinearKernel()),
                LinearLayer(3, 5, kernel=nn_kernel.LinearKernel()),
                SigmoidLayer(5, 3, kernel=nn_kernel.LinearKernel()),
                LinearLayer(3, 5, kernel=nn_kernel.LinearKernel()),
                SoftArgMaxCrossEntropy(5, 3, kernel=nn_kernel.LinearKernel())]
    
    net = nnImpl(layers, epochs=10 ** 4, speed=10 ** -4)
    
    l = 10000
    # ev = [ [n] for n in range(2, l, 2)]
    # y_ev = [[1, 0]] * len(ev)
    
    # odd = [ [n] for n in range(1, l, 2)]
    # y_odd = [[0, 1]] * len(odd)
    
    # x_ = ev + odd
    # y_ = y_ev + y_odd
    
    from numpy.random import randint
    from numpy.random import rand
    x_ = np.array([[rand(), rand(), rand() ] 
                   for _ in range(l)])
    
    y_ = np.array([[1, 0, 0] if x[2] < 0.33 else [0, 1, 0] if x[2] < 0.66 else [0, 0, 1]
                   for x in x_])

    net.fit(x_, y_)
    
    res = net.predict(np.array([[0.1, 0.2, 0.3]]))

    best = res.argmax(axis=1)
    print(f'Prediction is {res}, best={best}')


import torch.nn as nn
import torch
import torch.autograd as grad


def test_torch():

    x = torch.tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True, dtype=torch.float64)

    softmax = nn.Softmax(0)

    y = softmax(x)
    print(y)
    res = grad.grad(y, x)

    print(res)
    pass


if __name__ == '__main__':
    test()
    pass
