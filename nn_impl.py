
from  nn_layer import nnLayer
import nn_functions as funs

import numpy as np


class nnImpl:

    def __init__(self, layers: np.ndarray[nnLayer], in_dim: int, out_dim: int, epochs: int, speed,
                 loss_op):
        self.layers: np.ndarray[nnLayer] = layers

        self.epochs = epochs
        self.speed = speed
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.loss_op = loss_op

        pass

    def forward(self, x_mat: np.ndarray):
        for layer in self.layers:
            layer.X_mat = x_mat
            layer.Y_mat = np.dot(layer.X_mat, layer.W_mat)
            layer.Y_mat = np.add(layer.Y_mat, layer.B_vec)
            layer.S_mat = layer.activ_op(layer.Y_mat)
            x_mat = layer.S_mat 

    def count_grads(self, S_exp_mat: np.ndarray):
        self.count_grad_last_layer(S_exp_mat)
        for i in range(len(self.layers) - 2, -1, -1):
            self.count_grad_two_layers(self.layers[i], self.layers[i + 1])
            
    def backward(self):
        for layer in self.layers:
            self.apply_gradient_descent(layer)
    
    def fit(self, x_mat: np.ndarray, y_exp: np.ndarray):
        assert x_mat.shape[1] == self.in_dim
        assert y_exp.shape[1] == self.out_dim
        assert x_mat.shape[0] == y_exp.shape[0]

        for _ in range(self.epochs):
            self.forward(x_mat)
            self.count_grads(y_exp)
            self.backward()
        
    def predict(self, x_mat: np.ndarray):
        assert x_mat.shape[1] == self.in_dim
        self.forward(x_mat)
        return self.layers[-1].S_mat

    def count_grad_last_layer(self, S_exp_mat: np.ndarray):
        
        layer = self.layers[-1]
    
        prime_S_x_mat = layer.activ_prime_op(layer.Y_mat)

        grad_L_y_mat = self.loss_op(layer.S_mat, S_exp_mat)
        
        layer.grad_L_x_mat = np.multiply(prime_S_x_mat, grad_L_y_mat)

        layer.grad_L_w_mat = np.dot(layer.X_mat.transpose(), layer.grad_L_x_mat)
        
    def count_grad_two_layers(self, cur_layer: nnLayer, next_layer: nnLayer):

        buf = np.dot(next_layer.grad_L_x_mat, next_layer.W_mat.transpose())
        
        prime_S_x_mat = cur_layer.activ_prime_op(cur_layer.Y_mat)

        cur_layer.grad_L_x_mat = np.multiply(prime_S_x_mat, buf)

        cur_layer.grad_L_w_mat = np.dot(cur_layer.X_mat.transpose(), cur_layer.grad_L_x_mat)

    def apply_gradient_descent(self, layer: nnLayer):
        layer.W_mat = np.subtract(layer.W_mat, np.dot(self.speed, layer.grad_L_w_mat)) 
  
