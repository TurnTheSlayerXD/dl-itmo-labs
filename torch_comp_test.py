from numpy.random import random
import torch
from nn_grad import Node

import numpy as np


def gauss_T(lhs: torch.Tensor, rhs: torch.Tensor):

    D_sq = torch.norm(torch.unsqueeze(lhs, 1) - rhs.T, dim=2) ** 2
    res = (Node.q_dot * D_sq).exp()

    return res


def main():

    EPS = Node.EPS

    x0 = random((6, 2))  # 6 x 2
    w1 = random((2, 3))  # 2 x 3
    b1 = random(3)
    w2 = random((3, 4))  # 3 x 4
    b2 = random(4)
    t = random((6, 4))  # 6 x 4

    x0_n = Node(x0)

    w1_n = Node(w1)
    w2_n = Node(w2)
    b1_n = Node(b1)
    b2_n = Node(b2)

    t_n = Node(t)

    # z1 = x0_n.dot(w1_n) + b1
    epochs = 3
    step = 10 ** -2

    grads_node = []
    for i in range(epochs):
        z1 = x0_n.gauss(w1_n)
        # z1 = x0_n.dot(w1_n) + b1
        h = z1.sigmoid()

        print(f'h ={h}')

        z2 = h.gauss(w2_n)
        # z2 = h.dot(w2_n) + b2

        y = z2.sigmoid()

        res = y.softargmax()

        # error = res.minsqr_loss(t_n)
        error = res.log_loss(t_n)

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

    # print("b1.grad =", repr(b1_n.grad))
    # print("b2.grad =", repr(b2_n.grad))

    print("=========================")

    x0_t = torch.tensor(x0, requires_grad=True)
    w1_t = torch.tensor(w1, requires_grad=True)
    w2_t = torch.tensor(w2, requires_grad=True)
    b1_t = torch.tensor(b1, requires_grad=True)
    b2_t = torch.tensor(b2, requires_grad=True)

    t_t = torch.tensor(t, requires_grad=True)

    grads_torch = []
    for i in range(epochs):

        # z1 = x0_t.matmul(w1_t) + torch.tensor(b1)
        z1 = gauss_T(x0_t, w1_t)
        h = z1.sigmoid()
        h.retain_grad()

        # z2 = h.matmul(w2_t) +  torch.tensor(b2)
        z2 = gauss_T(h, w2_t)
        z2.retain_grad()

        y = z2.sigmoid()
        y.retain_grad()

        res = y.softmax(1)
        res.retain_grad()

        error = (-t_t * res.log()).sum(1).sum(0)

        error.backward(retain_graph=True, gradient=torch.ones(error.shape))
        print(f'error = {error}')
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
