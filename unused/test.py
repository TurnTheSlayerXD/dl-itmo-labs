import torch
from nn_grad import *


def gauss_T(lhs, rhs):
    q = 0.5
    q_dot = -1 / (2 * q ** 2)
    D_sq = (lhs.square()).sum(dim=1, keepdims=True)  \
        + (rhs.square()).sum(dim=0) \
        - 2 * lhs.matmul(rhs)
    res = (q_dot * D_sq).exp()

    return res


def net(in_dim, out_dim):
    from numpy.random import random

    w_ = np.array(random(in_dim, 3))
    b1 = random(3)

    w2 = random(3, 5)
    b2 = random(5)

    w3 = random(5, out_dim)
    b3 = random(out_dim)

    

    
def main():

    x0 = random((6, 2))  # 6 x 2
    w1 = random((2, 3))  # 2 x 3
    b1 = random(3)
    w2 = random((3, 4))  # 3 x 4
    b2 = random(4)
    t = random((6, 4))  # 6 x 4

    x0_n = Node(x0)

    w1_n = Node(w1)
    w2_n = Node(w2)

    t_n = Node(t)

    # z1 = x0_n.dot(w1_n) + b1
    z1 = x0_n.gauss(w1_n)

    h = z1.sigmoid()

    # z2 = h.dot(w2_n) + b2
    z2 = h.gauss(w2_n)

    y = z2.sigmoid()

    res = y.softargmax()

    error = res.minsqr_loss(t_n)

    error.backward(np.ones(error.shape))
    print(f'h grad = {h.grad}')
    print(f'z2 grad = {z2.grad}')
    print(f'y grad = {y.grad}')
    print(f'res = {res.grad}')
    print(f'error = {error}')
    print("w1.grad =", w1_n.grad)
    print("w2.grad =", w2_n.grad)
    print("=========================")

    x0_t = torch.tensor(x0, requires_grad=True)
    w1_t = torch.tensor(w1, requires_grad=True)
    w2_t = torch.tensor(w2, requires_grad=True)

    t_t = torch.tensor(t, requires_grad=True)

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

    error = (res - t_t).square().sum(dim=1).sum(dim=0)

    error.backward(retain_graph=True, gradient=torch.ones(error.shape))
    print(f'h grad = {h.grad}')
    print(f'z2 grad = {z2.grad}')
    print(f'y grad = {y.grad}')
    print(f'res = {res.grad}')
    print(f'error = {error}')
    print("w1.grad =", w1_t.grad)
    print("w2.grad =", w2_t.grad)

    EPS = 10 ** -5

    assert np.all(np.abs(w1_t.grad.numpy() - w1_n.grad) < EPS)


if __name__ == '__main__':

    main()
