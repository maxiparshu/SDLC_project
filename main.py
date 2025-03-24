# # # # # import cv2
# # # # # from kivy.app import App
# # # # # from kivy.uix.image import Image
# # # # # from kivy.clock import Clock
# # # # # from kivy.uix.boxlayout import BoxLayout
# # # # # from kivy.uix.button import Button
# # # # # from kivy.graphics.texture import Texture
# # # # # from plyer import storagepath
# # # # # import os
# # # # #
# # # # # class CameraApp(App):
# # # # #     def build(self):
# # # # #         self.capture = cv2.VideoCapture(1)  # Открываем камеру
# # # # #         self.image_widget = Image()  # Виджет для отображения изображения
# # # # #
# # # # #         # Создаем макет
# # # # #         layout = BoxLayout(orientation='vertical')
# # # # #         layout.add_widget(self.image_widget)
# # # # #
# # # # #         # Кнопка для захвата изображения
# # # # #         btn_capture = Button(text='Сделать снимок', size_hint_y=None, height=50)
# # # # #         btn_capture.bind(on_press=self.capture_image)
# # # # #         layout.add_widget(btn_capture)
# # # # #
# # # # #         # Запускаем таймер для обновления кадра
# # # # #         Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS
# # # # #
# # # # #         return layout
# # # # #
# # # # #     def update(self, dt):
# # # # #         ret, frame = self.capture.read()  # Читаем кадр с камеры
# # # # #         if ret:
# # # # #             # Преобразуем цвет из BGR (OpenCV) в RGB (Kivy)
# # # # #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # # # #             # Получаем размеры кадра
# # # # #             h, w = frame.shape[:2]
# # # # #             # Создаем текстуру (если она еще не создана)
# # # # #             texture = Texture.create(size=(w, h), colorfmt='rgb')
# # # # #             texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
# # # # #             texture.flip_vertical()  # Переворачиваем изображение по вертикали
# # # # #             self.image_widget.texture = texture
# # # # #
# # # # #     def capture_image(self, instance):
# # # # #         ret, frame = self.capture.read()  # Читаем кадр с камеры
# # # # #         if ret:
# # # # #             # Получаем путь к папке "Pictures"
# # # # #             pictures_folder = storagepath.get_pictures_dir()
# # # # #             if pictures_folder:
# # # # #                 # Создаем путь к файлу
# # # # #                 filepath = os.path.join(pictures_folder, "screenshot.png")
# # # # #                 # Сохраняем изображение
# # # # #                 cv2.imwrite(filepath, frame)
# # # # #                 print(f"Изображение сохранено как {filepath}")
# # # # #             else:
# # # # #                 print("Не удалось получить доступ к папке 'Pictures'")
# # # # #
# # # # #     def on_stop(self):
# # # # #         self.capture.release()  # Закрываем камеру при выходе из приложения
# # # # # й
# # # # # if __name__ == '__main__':
# # # # #     CameraApp().run()
# import cv2
# import mediapipe as mp
# import numpy as np
# import math
#
# from datasets import load_data, data_len
# from neural import Neural, to_full
#
#
# def find_similar_pixels(img, center, tolerance):
#     lower_bound = np.clip(center - tolerance, 0, 255)
#     upper_bound = np.clip(center + tolerance, 0, 255)
#     return cv2.inRange(img, lower_bound, upper_bound)
#
#
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
# counter = 212
# folder = "data/C"
# img_size = 250
#
# neural = Neural(img_size ** 2, data_len())
# # epochs = 1200
# # neural.train(epochs)
# # exit(228)
# neural.load_model()
# # (_, _), (x_test, y_test), names = load_data()
# #
# # for i in range(len(x_test)):
# #     x = x_test[i]
# #     y = y_test[i]
# #     print(names[y])
# #     f = open("check.txt", "w")
# #     for z in x:
# #         f.write(f"{z}".replace("\n", "") + "\n")
# #     f.close()
# #     print(to_full(y, 4))
# #     print(neural.predict(x))
# #     print(neural.predict_with_name(x))
# #     input(">>>")
#
# name = "hand"
# with mp_hands.Hands(
#         model_complexity=1,
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.7) as hands:
#     cap = cv2.VideoCapture(0)
#     hand_window_open = False
#
#     while cap.isOpened():
#         margin = 20
#         hand_pos_x = 0
#         hand_pos_y = 0
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             # If loading a video, use 'break' instead of 'continue'.
#             continue
#
#         # To improve performance, optionally mark the image as not writeable to
#         # pass by reference.
#         image = cv2.flip(image, 1)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = hands.process(image)
#
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         image_rect = image.copy()
#         height, width, _ = image.shape
#         # Draw the hand annotations on the image.
#
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 x_min = width
#                 y_min = height
#                 x_max = 0
#                 y_max = 0
#                 for landmark in hand_landmarks.landmark:
#                     # Преобразуем нормализованные координаты в пиксельные
#                     x = int(landmark.x * width)
#                     y = int(landmark.y * height)
#
#                     # Обновляем границы прямоугольника
#                     x_min = min(x_min, x)
#                     y_min = min(y_min, y)
#                     x_max = max(x_max, x)
#                     y_max = max(y_max, y)
#                 hand_pos_y = y_max
#                 hand_pos_x = x_max
#                 x_min = max(0, x_min - margin)
#                 y_min = max(0, y_min - margin)
#                 x_max = min(width, x_max + margin)
#                 y_max = min(height, y_max + margin)
#
#                 hand_region = image[y_min:y_max, x_min:x_max]
#                 cv2.rectangle(image_rect, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#
#                 img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
#                 hand_detect = img_white
#                 hand_shape = hand_region.shape
#                 hand_height = hand_shape[0]
#                 hand_width = hand_shape[1]
#
#                 if hand_region.size > 0:
#                     aspect = hand_height / hand_width
#
#                     hand_resize = hand_region.copy()
#
#                     if aspect > 1:
#                         k = img_size / hand_height
#                         width_calc = math.ceil(k * hand_width)
#                         hand_resize = cv2.resize(hand_region, (width_calc, img_size))
#                         width_gap = math.ceil((img_size - width_calc) / 2)
#                         hand_resize_cropped = hand_resize[:250, :250]
#                         img_white[:, width_gap: width_gap + width_calc] = hand_resize_cropped
#                     else:
#                         k = img_size / hand_width
#                         height_calc = math.ceil(k * hand_height)
#                         hand_resize = cv2.resize(hand_region, (img_size, height_calc))
#                         height_gap = math.ceil((img_size - height_calc) / 2)
#                         hand_resize_cropped = hand_resize[:250, :250]
#                         img_white[height_gap: height_gap + height_calc, :] = hand_resize_cropped
#
#                     hei, wid, _ = img_white.shape
#
#                     # Координаты центрального пикселя
#                     for attempt in range(1, 21):
#                         # Выбор центрального пикселя
#                         y = hei // 2 + (attempt // 3)  # Смещение для новых пикселей
#                         x = wid // 2 + (attempt % 3)
#                         y = np.clip(y, 0, hei - 1)
#                         x = np.clip(x, 0, wid - 1)
#
#                         center_pixel = img_white[y, x]
#
#                         mask = find_similar_pixels(img_white, center_pixel, 25)
#
#                         if not np.all(mask == 0):  # Если не полностью белое
#                             break
#
#
#
#                     # Создаем изображение с удалением фона
#                     output = np.zeros_like(img_white)
#                     output[mask > 0] = [0, 0, 0]
#                     result = cv2.bitwise_not(mask)
#
#                     # Отображение результата
#                     cv2.imshow('Result', result)
#                     binary_matrix = np.where(result == 255, 0, 1)
#                     name = neural.predict_with_name(binary_matrix)
#                     hand_window_open = True
#
#         else:
#             if hand_window_open:
#                 cv2.destroyWindow("Result")
#                 hand_window_open = False
#
#         image_final = cv2.flip(image_rect, 1)
#         if hand_window_open:
#             text_position = (width - hand_pos_x, hand_pos_y)
#             cv2.putText(image_final, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.imshow('Image', image_final)
#
#         key = cv2.waitKey(5)
#
#         if key & 0xFF == 27:
#             break
#         if key == ord("s"):
#             counter += 1
#             cv2.imwrite(f'{folder}/{counter}.jpg', result)
#             print(counter)
# cap.release()
# cv2.destroyAllWindows()
# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("riondsilva21/hand-keypoint-dataset-26k")
#
# print("Path to dataset files:", path)
import cv2
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
import os


class HandKeypointsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotations_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        print(img_path)
        # Открываем изображение
        image = Image.open(img_path).convert("RGB")

        # Ищем аннотацию для этого изображения
        annotation = next(ann for ann in self.annotations['annotations'] if ann['image_id'] == img_info['id'])
        keypoints = annotation['keypoints']

        # Преобразование изображения (если нужно)
        if self.transform:
            image = self.transform(image)

        # Преобразуем keypoints в тензор
        keypoints = torch.tensor(keypoints).view(-1, 3)  # (x, y, visibility)

        return image, keypoints


class HandKeypointModel(nn.Module):
    def __init__(self, num_keypoints=21):
        super(HandKeypointModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_keypoints * 3)

    def forward(self, x):
        return self.backbone(x)


def draw_keypoints(image, keypoints):
    for (x, y, z) in keypoints:
        if z < 2:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

base_path = '/datasets/riondsilva21/hand-keypoint-dataset-26k/versions/3/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/'

dataset = HandKeypointsDataset(
    annotations_file=base_path + 'coco_annotation/train/_annotations.coco.json',
    img_dir=base_path + 'images/train',
    transform=transform)
dataloader = DataLoader(dataset, batch_size=28, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = HandKeypointModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

# Цикл обучения
with open('logs.txt', 'w') as f:
    for epoch in range(num_epochs):
        model.train()  # Режим обучения
        running_loss = 0.0

        for images, keypoints in dataloader:
            # Перенос данных на устройство
            images = images.to(device)
            keypoints = keypoints.view(keypoints.size(0), -1).to(device)
            keypoints = keypoints.float()

            # Прямой проход
            outputs = model(images)
            outputs = outputs.float()

            # Вычисление ошибки
            loss = criterion(outputs, keypoints)

            # Обратный проход и обновление весов
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Сохраняем обученную модель
torch.save(model.state_dict(), 'hand_keypoints_model.pth')
#
# model = HandKeypointModel()
# model.load_state_dict(torch.load('hand_keypoints_model.pth'))
# model.to(device)
# model.eval()
#
# input_folder = base_path + "imagse/val"
# # Обрабатываем каждое изображение в папке
# for filename in os.listdir(input_folder):
#     if filename.endswith((".jpg", ".png", ".jpeg")):
#         # Загружаем изображение
#         img_path = os.path.join(input_folder, filename)
#         image = cv2.imread(img_path)
#
#         # Подготовка изображения для модели
#         input_image = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
#
#         # Пропускаем через модель
#         with torch.no_grad():
#             outputs = model(input_image).view(-1, 3).cpu().numpy()
#
#         # Масштабируем точки обратно к размеру изображения
#         h, w, _ = image.shape
#         outputs[:, 0] *= w / 224
#         outputs[:, 1] *= h / 224
#
#         # Копия для рисования ключевых точек
#         output_image = image.copy()
#         draw_keypoints(output_image, outputs)
#
#         # Объединяем оригинал и результат в одно изображение
#         combined_image = np.hstack((image, output_image))
#
#         # Показываем результат
#         cv2.imshow('Original vs Keypoints', combined_image)
#
#         # Ждем нажатие "q" или переход к следующему фото через 2 секунды
#         if cv2.waitKey(2000) & 0xFF == ord('q'):
#             break
#
# # Закрываем окно
# cv2.destroyAllWindows()
#
# print("✅ Обработка завершена!")

