from numpy.random import random
import torch
from nn_grad import Node

import numpy as np
import torch.nn.functional as F


def gauss_T(lhs: torch.Tensor, rhs: torch.Tensor):

    D_sq = torch.norm(torch.unsqueeze(lhs, 1) - rhs.T, dim=2) ** 2
    res = (Node.q_dot * D_sq).exp()

    return res


def main():

    EPS = Node.EPS

    from numpy.random import randint

    x0 = random((6, 2))  # 6 x 2
    w1 = random((2, 3))  # 2 x 3
    w2 = random((3, 4))  # 3 x 4

    t = np.array([[0., 0., 0., 1.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 1., 0., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])  # 6 x 4


    x0_n = Node(x0)

    w1_n = Node(w1)
    w2_n = Node(w2)

    t_n = Node(t)

    # z1 = x0_n.dot(w1_n) + b1
    epochs = 2
    step = 10 ** -2

    grads_node = []
    for i in range(epochs):
        z1 = x0_n.dot(w1_n)
        # z1 = x0_n.dot(w1_n) + b1
        h = z1.sigmoid()

        z2 = h.dot(w2_n)
        # z2 = h.dot(w2_n) + b2

        y = z2.sigmoid()



        res = y.softargmax()
        error = res.log_loss(t_n)

        # error = res.minsqr_loss(t_n)

        error.backward(np.ones(error.shape))

        print(f'error = {error}')

        print("w1.grad =", repr(w1_n.grad))
        print("w2.grad =", repr(w2_n.grad))

        grads_node.append(w1_n.grad)
        grads_node.append(w2_n.grad)

        w1_n.val = w1_n.val - w1_n.grad * step
        w2_n.val = w2_n.val - w2_n.grad * step
        w1_n.grad = None
        w2_n.grad = None

    print("============================================")

    for i in range(epochs):
        z1 = x0_n.dot(w1_n)
        # z1 = x0_n.dot(w1_n) + b1
        h = z1.sigmoid()

        z2 = h.dot(w2_n)
        # z2 = h.dot(w2_n) + b2

        y = z2.sigmoid()
        # error = res.minsqr_loss(t_n)

        error2 = y.crossentropy_loss(t_n)

        error2.backward(np.ones(error2.shape))

        print(f'error2 = {error2}')

        print("w1.grad =", repr(w1_n.grad))
        print("w2.grad =", repr(w2_n.grad))

        w1_n.val = w1_n.val - w1_n.grad * step
        w2_n.val = w2_n.val - w2_n.grad * step
        w1_n.grad = None
        w2_n.grad = None
    print("============================================")

    x0_t = torch.tensor(x0, requires_grad=True)
    w1_t = torch.tensor(w1, requires_grad=True)
    w2_t = torch.tensor(w2, requires_grad=True)

    t_t = torch.tensor(t, requires_grad=True)


    grads_torch = []
    for i in range(epochs):

        z1 = x0_t.matmul(w1_t)
        h = z1.sigmoid()
        h.retain_grad()

        z2 = h.matmul(w2_t)
        z2.retain_grad()

        y = z2.sigmoid()
        y.retain_grad()


        print(y.shape)
        print(t_t.shape)

        error1 = F.cross_entropy(y, t_t, reduction='sum')
        error1.backward(retain_graph=True,
                         gradient=torch.ones(error1.shape))

        print(f'error = {error1}')
        
        print("w1.grad =", w1_t.grad.numpy())
        print("w2.grad =", w2_t.grad)

        grads_torch.append(w1_t.grad.numpy() + 0)
        grads_torch.append(w2_t.grad.numpy() + 0)

        w1_t = w1_t - w1_t.grad * step
        w1_t.retain_grad()
        w2_t = w2_t - w2_t.grad * step
        w2_t.retain_grad()

    EPS = 10 ** -5

    for i in range(len(grads_node)):
        assert np.all(np.abs(grads_torch[i] - grads_node[i]) < EPS)

if __name__ == '__main__':

    main()
