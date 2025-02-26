import numpy as np
from numpy.random import shuffle

import nn_functions as funs

from nn_builder import nnBuilder


# proof of work
def test():

    builder = nnBuilder(in_dim=2, out_dim=1)

    builder.add_layer(4)
    builder.add_layer(1)
    

    net = builder.build()

    x_ = np.array([[0, 1], [1, 2], [3, 4]])
    y_ = np.array([[30.1], [-100], [100]])

    net.forward(x_)
    


if __name__ == '__main__':
    test()
    pass
