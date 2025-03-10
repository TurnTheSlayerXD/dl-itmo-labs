import numpy as np
from abc import abstractmethod, ABC
import nn_loss

import nn_kernel

from nn_loss import nnfix

from nn_grad import *


@abstractmethod
class nnLayer(ABC):

    def __init__(self, in_dim: int, out_dim: int, with_bias=True,
                 is_rand: bool = True, kernel=Node.dot):
        self.w_: Node = Node(np.random.rand(in_dim, out_dim)) \
            if is_rand \
            else Node(np.ones(shape=[in_dim, out_dim]))

        self.b_: Node = Node(np.random.rand(1, out_dim)) \
            if is_rand \
            else Node(np.zeros(shape=(1, out_dim)))

        self.kernel = kernel
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.forward = self.forward_w_bias if with_bias else self.forward_wout_bias
        self.descent = self.descent_w_bias if with_bias else self.descent_wout_bias

        self.x_: Node = None
        self.y_: Node = None

    @abstractmethod
    def forward_w_bias(self, x_: Node) -> Node:
        self.x_ = x_
        self.y_ = self.kernel(x_, self.w_) + self.b_
        return self.y_

    @abstractmethod
    def forward_wout_bias(self, x_: Node) -> Node:
        self.x_ = x_
        self.y_ = self.kernel(x_, self.w_)
        return self.y_

    def descent_w_bias(self, step):
        self.w_.val -= step * self.w_.grad
        self.b_.val -= step * self.b_.grad

    def descent_wout_bias(self, step):
        self.w_ -= step * self.w_.grad


class SigmoidLayer(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_w_bias(self, x_: Node) -> Node:
        super().forward_w_bias(x_)
        self.y_ = self.y_.sigmoid()
        return self.y_

    def forward_wout_bias(self, x_: Node) -> Node:
        super().forward_wout_bias(x_)
        self.y_ = self.y_.sigmoid()
        return self.y_


class TanhLayer(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_w_bias(self, x_: Node) -> Node:
        super().forward_w_bias(x_)
        self.y_ = self.y_.tanh()
        return self.y_

    def forward_wout_bias(self, x_: Node) -> Node:
        super().forward_wout_bias(x_)
        self.y_ = self.y_.tanh()
        return self.y_


class ReluLayer(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_w_bias(self, x_: Node) -> Node:
        super().forward_w_bias(x_)
        self.y_ = self.y_.relu()
        return self.y_

    def forward_wout_bias(self, x_: Node) -> Node:
        super().forward_wout_bias(x_)
        self.y_ = self.y_.relu()
        return self.y_


class SoftargmaxLayer(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_w_bias(self, x_: Node) -> Node:
        super().forward_w_bias(x_)
        self.y_ = self.y_.softargmax()
        return self.y_

    def forward_wout_bias(self, x_: Node) -> Node:
        super().forward_wout_bias(x_)
        self.y_ = self.y_.softargmax()
        return self.y_
