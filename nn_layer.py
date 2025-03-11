import numpy as np
from abc import abstractmethod, ABC



from nn_grad import *


@abstractmethod
class nnLayer(ABC):

    def __init__(self, in_dim: int, out_dim: int, with_bias=False,
                 is_rand: bool = True, kernel=Node.gauss):
        self.w_: Node = Node(np.random.rand(in_dim, out_dim)) \
            if is_rand \
            else Node(np.ones(shape=[in_dim, out_dim]))

        self.b_: Node = Node(np.random.rand(out_dim)) \
            if is_rand \
            else Node(np.zeros(shape=out_dim))

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
        
        self.w_.grad = None
        self.b_.grad = None

    def descent_wout_bias(self, step):
        self.w_.val -= step * self.w_.grad
        self.w_.grad = None


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

class LinearLayer(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_w_bias(self, x_: Node) -> Node:
        return super().forward_w_bias(x_)

    def forward_wout_bias(self, x_: Node) -> Node:
        return super().forward_wout_bias(x_)
