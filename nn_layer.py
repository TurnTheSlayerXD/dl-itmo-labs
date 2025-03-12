import numpy as np
from abc import abstractmethod, ABC


from nn_grad import *


@abstractmethod
class nnLayer(ABC):

    def __init__(self):
        super().__init__()

        self.x_: Node = None
        self.y_: Node = None

    @abstractmethod
    def forward(self, x_: Node) -> Node:
        raise NotImplemented


class SigmoidLayer(nnLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x_: Node) -> Node:
        return x_.sigmoid()


class TanhLayer(nnLayer):

    def __init__(self):
        pass

    def forward(self, x_: Node) -> Node:
        return x_.tanh()


class ReluLayer(nnLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x_: Node) -> Node:
        return x_.relu()


class SoftargmaxLayer(nnLayer):

    def __init__(self):
        pass

    def forward(self, x_: Node) -> Node:
        return x_.softargmax()


class LinearLayer(nnLayer):

    def __init__(self, in_dim: int, out_dim: int, with_bias: bool,
                 is_rand: bool = True, kernel=Node.dot):
        self.w_: Node = Node(np.random.rand(in_dim, out_dim)) \
            if is_rand \
            else Node(np.ones(shape=[in_dim, out_dim]))

        self.b_: Node = Node(np.random.rand(out_dim)) \
            if is_rand \
            else Node(np.zeros(shape=out_dim))

        self.kernel = kernel
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.forward_ = self.forward_w_bias if with_bias else self.forward_wout_bias

    def forward_w_bias(self, x_: Node) -> Node:
        return x_.dot(self.w_) + self.b_

    def forward_wout_bias(self, x_: Node) -> Node:
        return x_.dot(self.w_)

    def forward(self, x_: Node) -> Node:
        return self.forward_(x_)
