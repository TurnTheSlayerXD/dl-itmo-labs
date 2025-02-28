import numpy as np
from numpy.random import shuffle

import nn_functions as funs

from nn_builder import nnBuilder


# proof of work
def test():

    builder = nnBuilder(in_dim=2, out_dim=2, is_rand=True, epochs=10**4, speed=10 ** -4,
                        loss_op=funs.log_loss_prime)

    builder.add_layer(2, activ_op=funs.softmax,
                      activ_prime_op=funs.softmax_prime)

    net = builder.build()

    x_ = np.array([[-1, 2], [2, 3], [-3, 4]])
    y_ = np.array([[0, 1], [1, 0], [0, 1]])

    net.fit(x_, y_)
    
    res = net.predict(np.array([[2, 3]]))

    print(f'Prediction is {res}')


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
