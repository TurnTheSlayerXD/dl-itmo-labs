import numpy as np
from abc import abstractmethod, ABC


class nnKernel(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def dot(self, x_: np.ndarray, w_: np.ndarray):
        pass

    @abstractmethod
    def derivative(self, x_: np.ndarray, w_: np.ndarray, y_: np.ndarray | None=None):
        pass


class GausKernel(nnKernel):
    
    def __init__(self, q: float=1):
        self.q = q
        self.q_dot = -1 / (2 * self.q ** 2)
        self.q_prime = 1 / self.q ** 2
        pass

    def dot(self, x_: np.ndarray, w_: np.ndarray):
        D_sq = np.sum(np.square(x_) , axis=1, keepdims=True) \
                                    +np.sum(np.square(w_) , axis=0) \
                                    -2 * np.dot(x_, w_)   
        res = np.exp(self.q_dot * D_sq)                         
        return res

    def derivative(self, x_: np.ndarray, w_: np.ndarray, y_: np.ndarray | None=None):
        if y_ is None:
            y_ = self.dot(x_, w_)
        diff = x_[:, np.newaxis, :] - w_.T[np.newaxis, :, :]
        grad = y_[..., np.newaxis] * diff 
        grad = np.sum(grad, axis=1) * self.q_prime  
        return grad        

    
class LinearKernel(nnKernel):

    def __init__(self):
        pass

    def dot(self, x_: np.ndarray, w_: np.ndarray): 
        return np.dot(x_, w_)

    def derivative(self, x_: np.ndarray, w_: np.ndarray, y_: np.ndarray | None=None):
        return x_
        
