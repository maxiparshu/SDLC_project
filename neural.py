import numpy as np

from activation import softmax, relu, relu_derivative, softmax_batch, tanh, tanh_derivative, sigmoid, sigmoid_derivative
from datasets import load_data
from dense import Dense, Dropout
from loss import CrossEntropy

from datetime import datetime


# Получение текущего времени


def to_full(y_, len_):
    temp = np.zeros(len_)
    temp[y_ - 1] = 1
    return temp


def to_full_batch(y_batch, len_):
    temp = np.zeros((len(y_batch), len_))

    # Преобразуем каждую метку в one-hot вектор
    for i, y_ in enumerate(y_batch):
        temp[i, y_ - 1] = 1
    return temp


f = open("logs.txt", "w+")


class Neural:

    def __init__(self, input_len, output_len, reg_lambda=0.01, dropout_p=0.4, learning_rate=0.075):
        hidden_len = 80
        self.output_len = output_len
        self.input_len = input_len
        self.layer1 = Dense(input_len, hidden_len, reg_lambda=reg_lambda)
        self.dropout = Dropout(p=dropout_p)
        self.output_layer = Dense(hidden_len, output_len, reg_lambda=reg_lambda)
        self.loss_function = CrossEntropy()
        self.learning_rate = learning_rate
        self.loss = 0
        (self.x_train, self.y_train), (_, _), self.names = load_data()

    def forward(self, x, train=True):
        z1 = self.layer1.forward(x)
        a1 = relu(z1)
        d1 = self.dropout.forward(a1, train=train)

        z2 = self.output_layer.forward(d1)
        return softmax_batch(z2), z1

    def backward(self, z1, batch_size):
        d_loss = self.loss_function.backward_batch()
        d_z2 = self.output_layer.backward(d_loss, self.learning_rate, mini_batch=True, len_mini_batch=batch_size)

        d_d1 = self.dropout.backward(d_z2)
        d_a1 = relu_derivative(z1) * d_d1
        self.layer1.backward(d_a1, self.learning_rate, mini_batch=True, len_mini_batch=batch_size)

    def predict(self, input_layer):
        x = np.array(input_layer).flatten()
        x = x.reshape(1, -1)
        z1 = self.layer1.forward(x)
        t1 = relu(z1)

        z2 = self.output_layer.forward(t1)

        return softmax(z2)

    def predict_with_name(self, input_layer):
        x = np.array(input_layer).flatten()
        x = x.reshape(1, -1)


        z1 = self.layer1.forward(x)
        t1 = relu(z1)

        t2 = self.output_layer.forward(t1)
        return self.names[np.argmax(softmax(t2)) + 1]

    def save_model(self):
        w1, b1 = self.layer1.get_weight()
        w2, b2 = self.output_layer.get_weight()
        p = self.dropout.get_p()
        np.savez("neural_network_model.npz", W1=w1, b1=b1, W2=w2, b2=b2, p=p)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Model saved to neural_network_model.npz|| {current_time}")

    def load_model(self):
        data = np.load("neural_network_model.npz")
        W1 = data['W1']
        b1 = data['b1']
        W2 = data['W2']
        b2 = data['b2']
        p = data['p']
        self.layer1.set_weight(W1, b1)
        self.output_layer.set_weight(W2, b2)
        self.dropout.set_p(p)

    def train(self, epochs=1000, batch_size=128):
        # Обучение сети
        for epoch in range(epochs):

            total_loss = 0
            count = 0
            from sklearn.utils import shuffle
            x_train, y_train = shuffle(self.x_train, self.y_train, random_state=42)
            batched_data = list(zip(x_train, y_train))
            for i in range(0, len(batched_data), batch_size):
                batch = batched_data[i:i + batch_size]
                x_batch, y_batch = zip(*batch)  # Разархивируем батч данных и меток

                # Преобразуем обратно в массивы
                x_batch = np.array(x_batch).reshape(len(x_batch), self.input_len)
                y_batch = np.array(y_batch)
                y_true = to_full_batch(y_batch, self.output_len)

                # Прямое распространение
                y_pred, z1 = self.forward(x_batch, train=True)

                # Вычисление потерь
                self.loss = self.loss_function.forward_batch(y_true, y_pred)
                total_loss += self.loss

                # Обратное распространение
                self.backward(z1, batch_size)
                print(f"c{epoch}, {count} , {self.loss},total = {total_loss}")
                count += 1

            if (epoch + 1) % 10 == 0:
                now = datetime.now()
                current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f'{self.x_train[0]}\n{self.predict(self.x_train[0])}\n{self.y_train[0]}\n')
                f.write(f"Epoch {epoch + 1}, Loss: {total_loss / len(x_train)} // {current_time}\n")
        self.save_model()
