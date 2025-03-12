import numpy as np

from nn_layer import *



def nnfix(arr: np.ndarray):
    isnan = np.isnan(arr)
    isinf = np.isinf(arr)
    isneginf = np.isneginf(arr)
    
    arr[np.abs(arr) < 10 ** -5] = np.random.rand() * 10 **-3
    arr[arr > 10 ** 6] =  np.random.rand() * 10 ** 6
    arr[arr < -10 ** 6] = np.random.rand() * -10 ** 6
    
    arr[isnan] = np.random.rand() * 10 ** -3
    arr[isinf] = 10 ** 6 * np.random.rand()
    arr[isneginf] = -10 ** 6 * np.random.rand()
    

    arr[np.abs(arr) < 10 ** -5] = 0
    arr[arr > 10 ** 6] = 10 ** 6
    arr[arr < -10 ** 6] = -10 ** 6

    arr[isnan] = 0
    arr[isinf] = 10 ** 6
    arr[isneginf] = -10 ** 6


class nnLoss(ABC):
    def __init__(self):
        self.loss_: Node = None

    @abstractmethod
    def backward(self) -> float:
        raise NotImplemented

    @abstractmethod
    def count_loss(self, predicted: Node, true: Node) -> Node:
        raise NotImplemented


class MinsqrLoss(nnLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, predicted: Node, true: Node) -> Node:
        self.loss_ = predicted.minsqr_loss(true)
        self.loss_.backward(np.ones(self.loss_.shape))
        return self.loss_

    def count_loss(self, predicted: Node, true: Node) -> Node:
        self.loss_ = predicted.minsqr_loss(true)
        return self.loss_


class LogLoss(nnLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, predicted: Node, true : Node) -> Node:
        self.loss_ = predicted.log_loss(true)
        self.loss_.backward(np.ones(self.loss_.shape))
        return self.loss_

    def count_loss(self, predicted: Node, true : Node) -> Node:
        self.loss_ = predicted.log_loss(true)
        return self.loss_

class CrossEntropyLoss(nnLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def backward(self, predicted: Node, true: Node) -> Node:
        self.loss_ = predicted.crossentropy_loss(true)
        self.loss_.backward(np.ones(self.loss_.shape))
        return self.loss_

    def count_loss(self, predicted: Node, true: Node) -> Node:
        self.loss_ = predicted.crossentropy_loss(true)
        return self.loss_