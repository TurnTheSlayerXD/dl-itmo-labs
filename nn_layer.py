import numpy as np
from abc import abstractmethod, ABC
import nn_loss

import nn_kernel


from nn_loss import nnfix
@abstractmethod
class nnLayer(ABC):

    def __init__(self, in_: int, out_: int,
                 is_rand: bool=True, kernel=nn_kernel.LinearKernel(),
                 loss=nn_loss.MinsqrLoss(), with_bias: bool=False):
        self.W_mat: np.ndarray = np.random.rand(in_, out_) \
            if is_rand \
            else np.ones(shape=[in_, out_])
        
        self.B_vec: np.ndarray = np.random.rand(1, out_) \
            if is_rand \
            else np.zeros(shape=(1, out_))
            
        self.in_: int = in_
        self.out_: int = out_

        self.X_mat: np.ndarray = None
        
        self.Y_mat: np.ndarray | None = None
        self.S_mat: np.ndarray = None 

        self.grad_L_y_mat: np.ndarray = None

        self.grad_L_w_mat: np.ndarray = None
        self.grad_L_b_vec: np.ndarray = None
        
        self.loss_val = None

        self.kernel = kernel
        self.loss = loss
        
        self.with_bias = with_bias
        self.forward = self.forward_with_bias if with_bias else self.forward_without_bias
        
        self.descent = self.descent_with_bias if with_bias else self.descent_without_bias
        
        def grad_by_expected_with_bias(S_exp_mat: np.ndarray):
            self.grads_by_expected(S_exp_mat)
            self.grads_bias()

        def grad_by_layer_with_bias(S_exp_mat: np.ndarray):
            self.grads_by_layer(S_exp_mat)
            self.grads_bias()
           
        self.grads_last = grad_by_expected_with_bias if with_bias else self.grads_by_expected
        self.grads_middle = grad_by_layer_with_bias if with_bias else self.grads_by_layer
        
    def forward_with_bias(self, x_mat):
            self.X_mat = x_mat
            self.Y_mat = self.kernel.dot(self.X_mat, self.W_mat)
            self.Y_mat = self.Y_mat + self.B_vec
            self.S_mat = self.activ_value(self.Y_mat)
            return self.S_mat
        
    def forward_without_bias(self, x_mat):
        self.X_mat = x_mat
        self.Y_mat = self.kernel.dot(self.X_mat, self.W_mat)
        self.S_mat = self.activ_value(self.Y_mat)
        return self.S_mat
    
    def grads_bias(self):
        ones = np.ones(shape=(1, self.grad_L_y_mat.shape[0]), dtype=np.float64)
        self.grad_L_b_vec = ones @ self.grad_L_y_mat
    
    def descent_without_bias(self, speed : float):
        nnfix(self.grad_L_w_mat)
        self.W_mat = self.W_mat - (speed * self.grad_L_w_mat) 
                
     
    def descent_with_bias(self, speed : float):
        
        self.W_mat = self.W_mat - (speed @ self.grad_L_w_mat) 
        self.B_vec = self.B_vec - (speed @ self.grad_L_b_vec) 
    
    @abstractmethod
    def activ_value(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def activ_grad(self, x, sigm_val=None) -> np.ndarray:
        pass
   
    @abstractmethod
    def grads_by_expected(self, S_exp_mat: np.ndarray):
        self.loss_val = self.loss.loss_val(self.S_mat, S_exp_mat)
    
        prime_S_y_mat = self.activ_grad(self.Y_mat, self.S_mat)

        grad_L_s_mat = self.loss.loss_prime(self.S_mat, S_exp_mat)
        
        self.grad_L_y_mat = prime_S_y_mat * grad_L_s_mat
        nnfix(self.grad_L_y_mat)
        
        grad_Y_w_mat = self.kernel.derivative(self.X_mat, self.W_mat, self.Y_mat)
        
        self.grad_L_w_mat = grad_Y_w_mat.transpose() @ self.grad_L_y_mat
    
    @abstractmethod
    def grads_by_layer(self, next_layer: 'nnLayer'):
        
        grad_S_y_mat = self.activ_grad(self.Y_mat, self.S_mat)
        
        grad_L_y2_dot_w2 = next_layer.grad_L_y_mat @ next_layer.W_mat.transpose()

        self.grad_L_y_mat = grad_S_y_mat * grad_L_y2_dot_w2
        
        grad_Y_w_mat = self.kernel.derivative(self.X_mat, self.W_mat, self.Y_mat)

        self.grad_L_w_mat = grad_Y_w_mat.transpose() @ self.grad_L_y_mat
    
    
class SigmoidLayer(nnLayer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def activ_value(self, x: np.ndarray) -> np.ndarray:
        ones = np.ones(shape=x.shape, dtype=np.float64)
                
        exp_x = np.exp(-x)
        nnfix(exp_x)
        res = ones / (ones + exp_x)
        return res

    def activ_grad(self, x, sigm_val=None) -> np.ndarray:
        ones = np.ones(shape=x.shape, dtype=np.float64)
        f = sigm_val if not sigm_val is None else self.activ_value(x)
        return np.multiply(f , (np.subtract(ones, f)))
     
    def grads_by_expected(self, S_exp_mat: np.ndarray): 
        super().grads_by_expected(S_exp_mat)

    def grads_by_layer(self, next_layer: 'nnLayer'):
        super().grads_by_layer(next_layer)


class TanhLayer(nnLayer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def activ_value(self, x) -> np.ndarray:
        EPS = 10 ** -10
        p_exp = np.exp(x)
        n_exp = np.ones(shape=x.shape) / (p_exp + EPS)
        return (p_exp - n_exp) / (p_exp + n_exp + EPS) 

    def activ_grad(self, x, tanh_val=None) -> np.ndarray:
        y = tanh_val if tanh_val != None else self.activ_value(x)
        return np.ones(shape=x.shape) - np.square(y) 
    
    def grads_by_expected(self, S_exp_mat: np.ndarray): 
        super().grads_by_expected(S_exp_mat)

    def grads_by_layer(self, next_layer: 'nnLayer'):
        super().grads_by_layer(next_layer)


class ReluLayer(nnLayer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def activ_value(self, x: np.ndarray) -> np.ndarray:
        res = np.copy(x)
        res[res < 0] = 0
        res[res >= 0] = x[res >= 0] 
        return res

    def activ_grad(self, x: np.ndarray, y=None) -> np.ndarray:
        res = np.copy(x)
        
        res[res < 0] = 0
        res[res >= 0] = 1 
        
        return res

    def grads_by_expected(self, S_exp_mat: np.ndarray): 
        super().grads_by_expected(S_exp_mat)

    def grads_by_layer(self, next_layer: 'nnLayer'):
        super().grads_by_layer(next_layer)


class LinearLayer(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def activ_value(self, x) -> np.ndarray:
        return x

    def activ_grad(self, x, y=None) -> np.ndarray:
        return np.ones(shape=x.shape)

    def grads_by_expected(self, S_exp_mat: np.ndarray): 
        super().grads_by_expected(S_exp_mat)

    def grads_by_layer(self, next_layer: 'nnLayer'):
        super().grads_by_layer(next_layer)


class SoftArgMaxCrossEntropy(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.loss = nn_loss.LogLoss()
    
    def activ_value(self, x: np.ndarray) -> np.ndarray:
        EPS = 10 ** -10 
        exp_x = np.exp(x)

        nnfix(exp_x)

        s = np.sum(exp_x, axis=1)
        s = np.repeat(s, x.shape[1], axis=0)
        s = np.reshape(s, exp_x.shape)

        res = np.divide(exp_x, s + EPS)
        return res 

    def activ_grad(self, x_, y_=None) -> np.ndarray:
        if y_ is None:
            y_ = self.activ_value(x_)
        # ones = np.ones(shape=x_.shape)
        # res = y_ * (ones - y_)
        
        ones = np.ones(shape=y_.shape[1])
        identity = np.identity(y_.shape[1])

        res = np.array([ (identity @ y * (ones - y)) - np.outer(y, y.transpose())  
                for y in y_])
        res = np.average(res, axis=1)
        assert res.shape == x_.shape
        return res
    
    def grads_by_expected(self, S_exp_mat: np.ndarray):
        self.loss_val = self.loss.loss_val(self.S_mat, S_exp_mat)
        
        self.grad_L_y_mat = self.Y_mat - S_exp_mat   

        grad_Y_w_mat = self.kernel.derivative(self.X_mat, self.W_mat, self.Y_mat)
        
        nnfix(grad_Y_w_mat)
        self.grad_L_w_mat = grad_Y_w_mat.transpose() @ self.grad_L_y_mat

    def grads_by_layer(self, next_layer: 'nnLayer'):
        super().grads_by_layer(next_layer)
    
