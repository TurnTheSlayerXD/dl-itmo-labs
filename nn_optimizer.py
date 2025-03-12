
from abc import abstractmethod, ABC



import numpy as np

from nn_layer import LinearLayer


class Optim(ABC):

    def __init__(self, net):
        super().__init__()
        self.with_bias = net.with_bias

        self.net = net
        self.lin_layers = list(
            filter(lambda l: isinstance(l, LinearLayer), net.layers))

        self.descend = self.descend_w_bias if self.with_bias else self.descend_wout_bias

    @abstractmethod
    def descend_w_bias():
        raise NotImplemented()

    @abstractmethod
    def descend_wout_bias():
        raise NotImplemented()


class StochasticOptim(Optim):
    def __init__(self, net, lr: float):
        super().__init__(net)
        self.lr = lr

    def descend_w_bias(self):
        for layer in self.lin_layers:
            layer.w_.val -= self.lr * layer.w_.grad
            layer.b_.val -= self.lr * layer.b_.grad

            layer.w_.grad = None
            layer.b_.grad = None

    def descend_wout_bias(self):
        for layer in self.lin_layers:
            layer.w_.val -= self.lr * layer.w_.grad
            layer.w_.grad = None


class AdamOptim(Optim):
    def __init__(self, net, lr: float):
        super().__init__(net)
        self.lr = lr

        self.V_w = [np.ones(l.w_.shape) for l in self.lin_layers]
        self.G_w = self.V_w.copy()

        self.V_b = [np.ones(l.b_.shape) for l in self.lin_layers]
        self.G_b = self.V_b.copy()

        self.b1 = 0.9
        self.b2 = 0.99
        self.eps = np.exp(-8)

    def descend_w_bias(self):
        for i, layer in enumerate(self.lin_layers):
            w_grad = layer.w_.grad
            b_grad = layer.b_.grad

            self.V_w[i] = self.b1 * self.V_w[i] + (1 - self.b1) * w_grad
            self.G_w[i] = self.b2 * self.G_w[i] + \
                (1 - self.b2) * np.square(w_grad)

            self.V_b[i] = self.b1 * self.V_b[i] + (1 - self.b1) * b_grad
            self.G_b[i] = self.b2 * self.G_b[i] + \
                (1 - self.b2) * np.square(b_grad)

            layer.w_.val -= self.lr / \
                np.sqrt(self.G_w[i] + self.eps) * self.V_w[i]
            layer.b_.val -= self.lr / \
                np.sqrt(self.G_b[i] + self.eps) * self.V_b[i]

            layer.w_.grad = None
            layer.b_.grad = None

    def descend_wout_bias(self):
        for i, layer in enumerate(self.lin_layers):
            w_grad = layer.w_.grad

            self.V_w[i] = self.b1 * self.V_w[i] + (1 - self.b1) * w_grad
            self.G_w[i] = self.b2 * self.G_w[i] + \
                (1 - self.b2) * np.square(w_grad)

            layer.w_.val -= self.lr / \
                np.sqrt(self.G_w[i] + self.eps) * self.V_w[i]

            layer.w_.grad = None
