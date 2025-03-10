import numpy as np

from nn_layer import *

from nn_impl import nnImpl


def nnfix(arr: np.ndarray):
    isnan = np.isnan(arr)
    isinf = np.isinf(arr)
    isneginf = np.isneginf(arr)

    arr[np.abs(arr) < 10 ** -5] = 0
    arr[arr > 10 ** 6] = 10 ** 6
    arr[arr < -10 ** 6] = -10 ** 6

    arr[isnan] = 0
    arr[isinf] = 10 ** 6
    arr[isneginf] = -10 ** 6


class nnLoss(ABC):
    def __init__(self, step=10 ** -4):
        self.loss_: Node = None
        self.step = step

    @abstractmethod
    def backward(self) -> float:
        pass


class MinsqrLoss(nnLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, predicted: Node, true: Node) -> Node:
        self.loss_ = predicted.minsqr_loss(true)
        self.loss_.backward(np.ones(self.loss_.shape))
        return self.loss_


class LogLoss(nnLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, predicted: Node, true: Node) -> Node:
        self.loss_ = predicted.log_loss(true)
        self.loss_.backward(np.ones(self.loss_.shape))
        return self.loss_
