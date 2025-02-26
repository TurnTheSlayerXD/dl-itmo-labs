
from  nn_layer import nnLayer
import nn_functions as funs

import numpy as np


class nnImpl:

    def __init__(self, layers: np.ndarray[nnLayer], in_dim: int, out_dim: int, epochs, speed):
        self.layers: np.ndarray[nnLayer] = layers

        self.epochs = epochs
        self.speed = speed
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        pass

    def forward(self, x_mat: np.ndarray):
        for layer in self.layers:
            x_mat = x_mat.transpose()

            layer.linear_out = np.dot(layer.weight_mat, x_mat).transpose()

            np.add(layer.linear_out, layer.bias_vec, out=layer.linear_out)

            layer.activ_out = layer.activ_op(layer.linear_out)

            print(f'shape = {layer.activ_out.shape}')
            x_mat = layer.activ_out 

    def backward(self, y_mat: np.ndarray):
        self.fix_last_layer(self.layers[-1], y_mat)
        for i in range(len(self.layers) - 2, -1, -1):
            self.fix_layer(self.layers[i], self.layers[i + 1])
        pass

    def make_dif(self, layer):
        for i in range(layer.n_neurons):
            delta_weight_vec = self.speed * layer.delta_vec[i] * layer.x_vec

            delta_weight_vec += 10 ** -5

            layer.weight_mat[i] -= delta_weight_vec

            delta_bias = self.speed * layer.delta_vec[i]

            delta_bias += 10 ** -5

            layer.bias_vec[i] -= delta_bias

        pass
   
    def fit_one(self, x_vec: np.ndarray, y_expected: np.ndarray):
        self.forward(x_vec)
        self.backward(y_expected)

    def fit(self, x_mat: np.ndarray, y_expected: np.ndarray):
        assert x_mat.shape[0] == y_expected.shape[0]
        assert (x_mat.shape[1] == self.x_len)

        if type(y_expected[0]) is not np.ndarray:
            y_expected = [np.array([y for _ in range(self.layers[-1].n_neurons)])
                          for y in y_expected]
        else:
            assert self.layers[-1].n_neurons == y_expected.shape[1]

        for epoch in range(self.epochs):
            # shuffle(x_mat)
            for i in range(len(x_mat)):
                self.fit_one(x_mat[i], y_expected[i])
            if epoch % 100 == 0:
                # print(f'Epoch: {epoch} : {self.layers[-1].weight_mat}')
                pass
        pass

    def predict(self, x_vec):
        self.forward(x_vec)

        return self.layers[-1].y_vec
