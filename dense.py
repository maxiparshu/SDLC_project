import numpy as np


class Dense:
    def __init__(self, input_neurons, output_neurons, reg_lambda=0.0):
        self.W = np.random.randn(input_neurons, output_neurons) * 0.01
        self.b = np.zeros(output_neurons)
        # Переменные для хранения градиентов
        self.dW = None
        self.db = None
        self.dx = None
        self.input_data = None
        self.reg_lambda = reg_lambda
        self.final_dW = 0
        self.final_db = 0

    def forward(self, x):
        self.x = x
        return self.x @ self.W + self.b

    def backward(self,
                 dE,
                 learning_rate=0.001,
                 mini_batch=False,
                 update=True,
                 len_mini_batch=None):
        self.dW = self.x.T @ dE
        self.db = np.sum(dE, axis=0)
        self.dx = dE @ self.W.T

        # Если регуляризация активна, добавляем регуляризационный член к градиенту весов
        if self.reg_lambda != 0:
            self.dW += self.reg_lambda * self.W

        if mini_batch:
            if not hasattr(self, 'final_dW'):
                self.final_dW = np.zeros_like(self.dW)
                self.final_db = np.zeros_like(self.db)

            self.final_dW += self.dW
            self.final_db += self.db

        # Если update=True, обновляем веса и смещения
        if update:
            if mini_batch and len_mini_batch:
                # Усредняем градиенты по размеру мини-батча перед обновлением
                self.W -= learning_rate * self.final_dW / len_mini_batch
                self.b -= learning_rate * self.final_db / len_mini_batch
                # Обнуляем аккумулированные градиенты после обновления
                self.final_dW = np.zeros_like(self.dW)
                self.final_db = np.zeros_like(self.db)
            else:
                # Обновляем веса и смещения без усреднения (для одного шага)
                self.W -= learning_rate * self.dW
                self.b -= learning_rate * self.db

        # Возвращаем градиенты по входам для следующего слоя в цепи
        return self.dx

    def get_weight(self):
        return self.W, self.b

    def set_weight(self, dw, db):
        self.W = dw
        self.b = db


class Dropout:
    def __init__(self, p=0.5):
        self.mask = None
        self.p = p

    def forward(self, x, train=True):
        if not train:
            self.mask = np.ones(x.shape)
            return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dz):
        return dz * self.mask

    def get_p(self):
        return self.p

    def set_p(self, new_p):
        self.p = new_p
