
from nn_layer import nnLayer
import numpy as np

from nn_loss import *


class nnImpl:

    def __init__(self, in_dim, out_dim, step = 10 ** -5):
        self.layers: np.ndarray[nnLayer] = [SigmoidLayer(in_dim, 3, ),
                                            LinearLayer(3, out_dim)]

        self.step = step
        
    def forward(self, x_: Node) -> Node:
        for layer in self.layers:
            x_ = layer.forward(x_)
        return x_

    
    
    def descent(self):
        for layer in self.layers:
            layer.descent(self.step)
