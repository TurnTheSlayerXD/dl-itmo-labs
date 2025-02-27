import numpy as np


class nnLayer:

    def __init__(self, n_neurons: int, in_dim: int,
                 is_rand: bool,
                 activ_op,
                 activ_prime_op):
        self.W_mat: np.ndarray = np.random.rand(in_dim, n_neurons) \
            if is_rand \
            else np.ones(shape=[in_dim, n_neurons])
        self.B_vec: np.ndarray = np.random.rand(n_neurons) \
            if is_rand \
            else np.zeros(n_neurons)

        self.n_neurons: int = n_neurons
        self.in_dim: int = in_dim

        self.X_mat: np.ndarray = None
        
        self.Y_mat: np.ndarray = None
        self.S_mat: np.ndarray = None 

        self.activ_op = activ_op
        self.activ_prime_op = activ_prime_op

        self.grad_L_x_mat: np.ndarray = None

        self.grad_L_w_mat: np.ndarray = None
