import numpy as np
import os


# его  номер
def load_data():  # x_train - двумерный массив каждый элемент это массив представляющий жест
    datasets = "data"
    folder_names = [name for name in os.listdir(datasets) if os.path.isdir(os.path.join(datasets, name))]

    x_train = []
    y_train = []
    y_train_name = []
    y_test = []
    x_test = []
    y_train_name.append("")
    for i in range(len(folder_names)):
        name = folder_names[i]
        y_train_name.append(name)
        for count in range(1, 346):
            y_train.append(i + 1)
            image_path = f'{datasets}/{name}/{count}.jpg'
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            binary_matrix = np.where(image == 255, 0, 1)
            x_train.append(binary_matrix)
        for count in range(346, 356):
            y_test.append(i + 1)
            image_path = f'{datasets}/{name}/{count}.jpg'
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            binary_matrix = np.where(image == 255, 0, 1)
            x_test.append(binary_matrix)

    from sklearn.utils import shuffle
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
    return (x_train, y_train), (x_test, y_test), y_train_name


def data_len():
    datasets = "data"
    return len([name for name in os.listdir(datasets) if os.path.isdir(os.path.join(datasets, name))])
