
import numpy as np
from abc import abstractmethod, ABC


class Node(ABC):

    q = 0.5
    q_dot = -1 / (2 * q ** 2)
    q_prime = 1 / q ** 2

    def __init__(self, val: np.ndarray | np.float64, op_name: str = 'none',
                 lhs: 'Node' = None,
                 rhs: 'Node' = None):

        self.val = np.array(val)
        self.op_name = op_name
        self.lhs = lhs
        self.rhs = rhs
        self.grad: np.ndarray = None

    def backward(self, grad: np.ndarray):

        if self.grad is None:
            self.grad = np.zeros(grad.shape, dtype=np.float64)

        self.grad += grad

        if self.op_name == '__add__':
            self.lhs.backward(grad)
            self.rhs.backward(grad)
        elif self.op_name == '__sub__':
            self.lhs.backward(grad)
            self.rhs.backward(-grad)
        elif self.op_name == '__mul__':
            self.lhs.backward(grad * self.rhs.val)
            self.rhs.backward(grad * self.lhs.val)
        elif self.op_name == '__truediv__':
            self.lhs.backward(grad / self.rhs.val)
            self.rhs.backward(-grad * self.lhs.val / np.square(self.rhs.val))
        elif self.op_name == 'dot':
            self.lhs.backward(np.dot(grad, self.rhs.val.T))
            self.rhs.backward(np.dot(self.lhs.val.T, grad))
        elif self.op_name == 'transpose':
            self.lhs.backward(grad.T)
        elif self.op_name == '__pow__':
            k = self.rhs.val
            self.lhs.backward(grad * k * self.lhs.val ** (k - 1))

        elif self.op_name == '__pos__':
            self.lhs.backward(grad)
        elif self.op_name == '__neg__':
            self.lhs.backward(-grad)

        elif self.op_name == 'exp':
            self.lhs.backward(grad * self.val)

        elif self.op_name == 'log':
            self.lhs.backward(grad / self.lhs.val)
        elif self.op_name == 'square':
            self.lhs.backward(2 * self.lhs.val * grad)

        elif self.op_name.startswith('sum'):
            dim = int(self.op_name.split('_')[1])
            if dim < grad.ndim:
                grad = np.sum(grad, axis=dim, keepdims=True)
            else:
                grad = np.expand_dims(grad, dim)
            grad = np.broadcast_to(grad, self.lhs.shape)
            self.lhs.backward(grad)

        elif self.op_name == 'sigmoid':
            self.lhs.backward(grad * self.val * (1 - self.val))

        elif self.op_name == 'tanh':
            self.lhs.backward(grad * (1 - np.square(self.val)))
        elif self.op_name == 'relu':
            res = np.copy(self.val)
            res[res < 0] = 0
            res[res >= 0] = 1
            self.lhs.backward(grad * res)
        elif self.op_name == 'softargmax':
            y_ = self.val
            eye = np.eye(y_.shape[1], dtype=np.float64)
            res = np.array([(eye * y - np.outer(y, y)) for y in y_])
            grad = np.expand_dims(grad, grad.ndim)
            fin = np.broadcast_to(grad, res.shape) * res
            fin = np.sum(fin, axis=1)
            self.lhs.backward(fin)
        elif self.op_name == 'gauss':
            y_ = self.val
            x_ = self.lhs.val
            w_ = self.rhs.val
            diff = np.expand_dims(x_, 1) - w_.T
            res = np.expand_dims(y_, 2) * diff * self.q_prime
            fin = np.expand_dims(grad, 2) * res
            self.lhs.backward(-np.sum(fin, 1))
            self.rhs.backward(np.sum(fin, 0).T)

        elif self.op_name == 'minsqr_loss':
            lhs = self.lhs.val
            rhs = self.rhs.val
            grad = np.broadcast_to(np.expand_dims(grad, (0, 1)), lhs.shape)
            res = grad * (lhs - rhs) * 2
            self.lhs.backward(res)
            self.rhs.backward(-res)
        elif self.op_name == 'log_loss':
            lhs = self.lhs.val
            rhs = self.rhs.val
            grad = -np.broadcast_to(np.expand_dims(grad, (0, 1)), lhs.shape)

            self.lhs.backward(grad * rhs / lhs)
            self.rhs.backward(grad / np.log(lhs))

    def __add__(self, rhs) -> 'Node':
        if not isinstance(rhs, Node):
            rhs = Node(rhs)
        return Node(self.val + rhs.val, '__add__', self, rhs)

    def __sub__(self, rhs) -> 'Node':
        if not isinstance(rhs, Node):
            rhs = Node(rhs)
        return Node(self.val - rhs.val, '__sub__', self, rhs)

    def __mul__(self, rhs) -> 'Node':
        if not isinstance(rhs, Node):
            rhs = Node(rhs)
        return Node(self.val * rhs.val, '__mul__', self, rhs)

    def __truediv__(self, rhs) -> 'Node':
        if not isinstance(rhs, Node):
            rhs = Node(rhs)
        return Node(self.val / rhs.val, '__truediv__', self, rhs)

    def __pow__(self, rhs) -> 'Node':
        if not isinstance(rhs, Node):
            rhs = Node(rhs)
        return Node(self.val ** rhs.val, '__pow__', self, rhs)

    def __pos__(self) -> 'Node':
        return Node(self.val, '__pos__', self)

    def __neg__(self) -> 'Node':
        return Node(-self.val, '__neg__', self)

    def square(self) -> 'Node':
        return Node(np.square(self.val), 'square', self)

    def dot(self, rhs) -> 'Node':
        if not isinstance(rhs, Node):
            rhs = Node(rhs)
        return Node(np.dot(self.val, rhs.val), 'dot', self, rhs)

    def exp(self) -> 'Node':
        return Node(np.exp(self.val), 'exp', self)

    def sum(self, dim, keepdims: bool = False) -> 'Node':
        name = f'sum_{dim}'
        return Node(np.sum(self.val, dim, keepdims=keepdims),
                    name, self)

    def sigmoid(self) -> 'Node':
        return Node(1 / (1 + np.exp(-self.val)), 'sigmoid', self)

    def tanh(self) -> 'Node':
        EPS = 10 ** -10
        p_exp = np.exp(self.val)
        n_exp = 1 / (p_exp + EPS)
        return Node((p_exp - n_exp) / (p_exp + n_exp), 'tanh', self)

    def log(self) -> 'Node':
        return Node(np.log(self.val), 'log', self)

    def relu(self) -> 'Node':
        res = np.copy(self.val)
        res[res < 0] = 0
        res_gt_zer = res >= 0
        res[res_gt_zer] = self.val[res_gt_zer]
        return Node(res, 'relu', self)

    def softargmax(self) -> 'Node':
        exp_x = np.exp(self.val)
        s = np.sum(exp_x, axis=1, keepdims=True)
        res = exp_x / s
        return Node(res, 'softargmax', self)

    def gauss(self, rhs) -> 'Node':
        if not isinstance(rhs, Node):
            rhs = Node(rhs)
        D_sq = np.sum(np.square(self.val), axis=1, keepdims=True) \
            + np.sum(np.square(rhs.val), axis=0) \
            - 2 * np.dot(self.val, rhs.val)
        res = np.exp(self.q_dot * D_sq)

        return Node(res, 'gauss', self, rhs)

    def log_loss(self, true) -> 'Node':
        if not isinstance(true, Node):
            true = Node(true)
        res = np.sum(np.sum(-(true.val * np.log(self.val)), axis=1),
                     axis=0)
        return Node(res, 'log_loss', self, true)

    def minsqr_loss(self, true) -> 'Node':
        if not isinstance(true, Node):
            true = Node(true)
        res = np.sum(np.sum((self.val - true.val) ** 2, axis=1),
                     axis=0)
        return Node(res, 'minsqr_loss', self, true)

    @property
    def T(self) -> 'Node':
        return Node(self.val.T, 'transpose', self)

    @property
    def shape(self) -> 'Node':
        return self.val.shape

    def __str__(self):
        return self.val.__str__()
