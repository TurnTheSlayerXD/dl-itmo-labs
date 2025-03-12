
from nn_layer import nnLayer
import numpy as np

from nn_loss import *


class nnImpl:

    def __init__(self, in_dim, out_dim):

        self.with_bias = True
        self.layers: np.ndarray[nnLayer] = [LinearLayer(in_dim, 4, with_bias=self.with_bias, kernel=Node.dot),
                                            ReluLayer(4, 3, with_bias=self.with_bias, kernel=Node.dot),
                                            SigmoidLayer(3, 4, with_bias=self.with_bias, kernel=Node.dot),
                                            LinearLayer(
                                                4, 5, with_bias=self.with_bias, kernel=Node.dot),
                                            SoftargmaxLayer(5, out_dim, with_bias=self.with_bias, kernel=Node.dot)]

    def forward(self, x_: Node) -> Node:
        for layer in self.layers:
            x_ = layer.forward(x_)
        return x_
