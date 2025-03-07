import numpy as np


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
    

class MinsqrLoss:

    def __init__(self):

        def minsqr_loss_value(pred, true) -> np.float64:
            return  np.sum(np.abs(pred - true) ** 2) / len(pred)

        def minsqr_loss_prime(pred, true) -> np.ndarray:
            res = 2 * (pred - true)
            return res
            
        self.loss_val = minsqr_loss_value
        self.loss_prime = minsqr_loss_prime
        
        
class LogLoss:

    def __init__(self):

        def log_loss_value(pred: np.ndarray, exp: np.ndarray) -> np.float64:
            EPS = 10 ** -10

            ones = np.ones(shape=pred.shape)

            res = -(exp * np.log2(pred + EPS) + (ones - exp) * np.log2(ones - pred + EPS))

            # res = -(exp * np.log2(pred + EPS))
            s = np.sum(res) / len(pred)
            
            return s

        def log_loss_prime(pred: np.ndarray, exp: np.ndarray) -> np.ndarray:
            EPS = 10 ** -10
            ones = np.ones(shape=pred.shape)
            res = (pred - exp) / ((pred + EPS) * (ones - pred + EPS))
            return res

        self.loss_val = log_loss_value
        self.loss_prime = log_loss_prime
        
