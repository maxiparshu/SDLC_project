import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(dz):
    dtanh = 1 - dz ** 2
    return dtanh


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    x_calc = x - np.max(x)
    p = np.exp(x_calc) / np.sum(np.exp(x_calc))
    return p


def softmax_batch(x):
    x_calc = x - np.max(x, axis=1, keepdims=True)

    exp_x = np.exp(x_calc)
    p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return p
