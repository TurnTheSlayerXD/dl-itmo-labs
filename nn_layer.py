import numpy as np


class nnLayer:

    def __init__(self, n_neurons: int, in_dim: int,
                 is_rand: bool, norm_op, norm_prime_op, error_prime_op):
        self.weight_mat = np.random.rand(n_neurons, in_dim) \
            if is_rand \
            else np.ones(shape=[n_neurons, in_dim])
        self.bias_vec = np.random.rand(n_neurons) \
            if is_rand \
            else np.zeros(n_neurons)

        self.n_neurons: int = n_neurons
        self.in_dim: int = in_dim

        self.linear_out: np.ndarray = None
        self.activ_out: np.ndarray = None
 
        self.delta_vec: np.ndarray = None
        self.x_vec: np.ndarray = None

        self.activ_op = np.vectorize(norm_op)
        self.activ_prime_op = np.vectorize(norm_prime_op)
        self.activ_prime_op = np.vectorize(error_prime_op)

