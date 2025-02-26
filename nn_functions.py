import numpy as np


def sigmoid(v):
    # return v
    return 1 / (1 + np.exp(-v))


def sigmoid_prime(v):
    return sigmoid(v) * (1 - sigmoid(v))


def min_sqr_prime(predicted, expected):
    return predicted - expected


def log_loss_prime(pred, true):
    if pred == 1:
        pred = 0.999
    return - true / pred + (1 - true) / (1 - pred)


def min_sqr_error_prime(pred, true):
    return pred - true


def no_norm_op(x):
    return x
