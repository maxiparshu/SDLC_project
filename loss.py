import numpy as np


class CrossEntropy:
    def __init__(self):
        self.loss = None
        self.y_true = None
        self.y_hat = None

    def forward(self, y_true, y_hat):
        self.y_hat = y_hat
        self.y_true = y_true
        self.loss = -np.sum(y_true * np.log(y_hat))
        if np.isinf(self.loss):
            return 0
        if self.loss != self.loss:
            return 0
        return self.loss

    def backward(self):
        # dz
        return self.y_hat-self.y_true

    def forward_batch(self, y_true, y_hat):
        epsilon = 1e-12
        self.y_hat = y_hat
        self.y_true = y_true
        y_hat_clipped = np.clip(y_hat, epsilon, 1. - epsilon)

        # Рассчитываем потери для каждого примера в батче и затем усредняем
        batch_loss = -np.sum(y_true * np.log(y_hat_clipped), axis=1)  # Потери для каждого примера
        self.loss = np.mean(batch_loss)  # Среднее по батчу
        if np.isinf(self.loss):
            return 0
        if self.loss != self.loss:
            return 0
        return self.loss

    def backward_batch(self):
        """
        Рассчитывает градиенты потерь по выходам сети для батча.
        Возвращает dL/dy_hat, где y_hat — предсказанные вероятности.
        """
        batch_size = self.y_true.shape[0]
        return (self.y_hat - self.y_true) / batch_size
