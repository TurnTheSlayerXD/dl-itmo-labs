

from  nn_layer import nnLayer
import nn_functions as funs

from nn_impl import nnImpl
import numpy as np


class nnBuilder:

    def __init__(self, in_dim: int, out_dim: int, is_rand=True, epochs=10 ** 4, speed=10 ** -4,
                 loss_op=funs.log_loss_prime):
        self.speed = speed
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.is_rand = is_rand
        self.layers = []
        self.epochs = epochs
        self.loss_op = loss_op

    def add_layer(self, n_neurons: int,
                  activ_op=funs.softmax,
                  activ_prime_op=funs.softmax_prime):
        assert n_neurons > 0
        
        layer = nnLayer(n_neurons, self.in_dim, self.is_rand, activ_op, activ_prime_op) \
            if len(self.layers) == 0 \
            else nnLayer(n_neurons, self.layers[-1].n_neurons,
                       self.is_rand, activ_op, activ_prime_op)

        self.layers.append(layer)

    def build(self) -> nnImpl:
        assert len(self.layers) > 0
        layers = np.array(self.layers)
        assert layers[-1].n_neurons == self.out_dim

        return nnImpl(np.array(self.layers), in_dim=self.in_dim, out_dim=self.out_dim,
                      epochs=self.epochs, speed=self.speed, loss_op=self.loss_op)
