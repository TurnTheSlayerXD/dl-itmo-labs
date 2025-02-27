import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    f = sigmoid(x)
    return f * (1 - f)


def linear(x):
    return x


def linear_prime(x):
    return 1


def relu(x):
    return 0 if x < 0 else x


def relu_prime(x):
    return 0 if x < 0 else 1


def tanh(x):
    p_exp = np.exp(x)
    n_exp = np.exp(-x)
    return (p_exp - n_exp) / (p_exp + n_exp) 


def tanh_prime(x):
    return 1 - tanh(x) ** 2


def softmax(x: np.ndarray):
    res = np.zeros(shape=x.shape)
    for i in range(len(x)):
        exp_x = np.exp(x[i])
        res[i] = exp_x / np.sum(exp_x)
        assert np.abs(np.sum(res[i]) - 1) < 0.0001
    return res


def softmax_prime(x):
    y = softmax(x)
    res = np.subtract(np.identity(len(y)), np.dot(y, y.transpose()))
    # print(f'x = {x}')
    # print(f'y = {y}')
    return res
    

def log_loss_prime(predicted, expected):
    return -expected / predicted + (1 - expected) / (1 - predicted)


def min_sqr_loss_prime(pred, true):
    return  2 * (pred - true)

