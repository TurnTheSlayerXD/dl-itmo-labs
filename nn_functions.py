import numpy as np


def sigmoid(x: np.ndarray):
    ones = np.ones(shape=x.shape)
    res = np.divide(ones, np.add(ones, np.exp(np.negative(x))))
    return res


def sigmoid_prime(x):
    ones = np.ones(shape=x.shape)
    f = sigmoid(x)
    return np.multiply(f , (np.subtract(ones, f)))


def linear(x):
    return x


def linear_prime(x):
    return np.ones(shape=x.shape)


def relu(x):
    return 0 if x < 0 else x


def relu_prime(x):
    return 0 if x < 0 else 1


def tanh(x):
    p_exp = np.exp(x)
    n_exp = np.divide(np.ones(shape=x.shape), p_exp)
    return np.divide(np.subtract(p_exp, n_exp), np.add(p_exp, n_exp)) 


def tanh_prime(x):
    return np.subtract(np.ones(shape=x.shape), np.square(tanh(x))) 


def softmax(x: np.ndarray):
    res = np.zeros(shape=x.shape)
    for i in range(len(x)):
        exp_x = np.exp(x[i])
        res[i] = exp_x / np.sum(exp_x)
        # assert np.abs(np.sum(res[i], axis=0) - 1) < 0.1
    return res


def softmax_prime(x):
    y = softmax(x)
    res = np.subtract(np.identity(len(y)), np.dot(y, y.transpose()))
    res = np.dot(res, x)
    # print(f'x = {x}')
    # print(f'y = {y}')
    return res
    

def log_loss_prime(predicted : np.ndarray, expected : np.ndarray):
    ones = np.ones(shape=predicted.shape)
    res = np.negative(np.add(np.divide(expected, predicted),
                           np.divide(np.subtract(ones, expected), np.subtract(ones, predicted)))) 
    return res


def min_sqr_loss_prime(pred, true):
    return  2 * (pred - true)

