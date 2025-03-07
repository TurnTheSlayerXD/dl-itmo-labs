
from  nn_layer import nnLayer
import numpy as np


class nnImpl:

    def __init__(self, layers: np.ndarray[nnLayer], epochs: int=10 ** 4, speed: float=10 ** -4):
        self.layers: np.ndarray[nnLayer] = layers

        self.epochs = epochs
        self.speed = speed

        self.loss_val: np.float = None
       
    def forward(self, x_mat: np.ndarray):
        for layer in self.layers:
            x_mat = layer.forward(x_mat)            

    def count_grads(self, S_exp_mat: np.ndarray):
        last = self.layers[-1] 
        last.grads_last(S_exp_mat)
        self.loss_val = last.loss_val
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].grads_middle(self.layers[i + 1])
            
    def backward(self):
        for layer in self.layers:
            layer.descent(self.speed)
    
    def fit(self, x_mat: np.ndarray, y_exp: np.ndarray):
        indxs = list(range(0, len(x_mat)))

        for epoch in range(1, self.epochs + 1):
            self.forward(x_mat)
            self.count_grads(y_exp)
            self.backward()
            if epoch % 10 ** 2 == 0:
                print(f'epoch: {epoch} loss value: {self.loss_val}')
                np.random.shuffle(indxs)
                x_mat = x_mat[indxs]
                y_exp = y_exp[indxs]

    def predict(self, x_mat: np.ndarray):
        self.forward(x_mat)
        return self.layers[-1].S_mat

    
