import sys
import glob
import cv2
import random
import numpy as np

import tflearn

from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing

import prepare_train_data

# нейтральный, злой, отвращение, страх, счастливый, грустный, удивленный
# EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def get_files(emotion):
    """
    Взять все изображения эмоции
    :param emotion:  название эмоции
    :return: 80% тренировочных и 20% тестовых изображений
    """

    files = glob.glob('datasets\\%s\\*' % emotion)
    random.shuffle(files)
    train = files[:int(len(files) * 0.8)]  # Берем первые 80% файлов эмоции для тренировки
    test = files[-int(len(files) * 0.2):]  # и последние 20% для тестирования
    return train, test


def make_sets():
    """
    Составить тренировочные и тестовые датасеты
    :return: набор тренировочных (изображения - эмоция) и тестовых
    """

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for emotion in EMOTIONS:
        train, test = get_files(emotion)
        for item in train:
            image = cv2.imread(item)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            train_data.append(gray_image)
            train_labels.append(EMOTIONS.index(emotion))

        for item in test:
            image = cv2.imread(item)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            test_data.append(gray_image)
            test_labels.append(EMOTIONS.index(emotion))

    train_data = np.reshape(train_data, [-1, 64, 64, 1])
    test_data = np.reshape(test_data, [-1, 64, 64, 1])

    return (train_data, train_labels), (test_data, test_labels)


# Вывод тех эмоций, на распознавание которых будем обучать нейросеть
print('----------')
print('Emotions for learning:', EMOTIONS)
print('----------')

# Подготовка тренировочных данных
aligned_file_count = prepare_train_data.preparing_data()
print('{} files aligned and saved in datasets'.format(aligned_file_count))
print('----------')


# Загрузить дата сет
(X, Y), (X_test, Y_test) = make_sets()

# Перемешать данные
X, Y = shuffle(X, Y)

# Преобразуем вектор выходных значение в двоичную матрицу для использования в categorical_crossentropy
Y = to_categorical(Y, len(EMOTIONS))
Y_test = to_categorical(Y_test, len(EMOTIONS))

# Для нормализации данных(масштабирование и отцентровка данных)
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Для создания дополнительных синтетический данных/увеличения кол-ва данных
# перевернутые, повернутые, размытые изображения
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.0)
img_aug.add_random_blur(sigma_max=3.0)

# Определяем архитектуру нейросети

# Входные изображения 64х64 в градациях серого
network = input_data(shape=[None, 64, 64, 1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Шаг первый - свертка
network = conv_2d(network, 64, 3, activation='relu')

# Шаг второй - пуллинг(уменьшение размерности)
network = max_pool_2d(network, 2)

# Шаг третий - повторная свертка
network = conv_2d(network, 128, 3, activation='relu')

# Шаг четвертый - повторная свертка
network = conv_2d(network, 128, 3, activation='relu')

# Шаг пятый - повторный пуллинг(уменьшение размерности)
network = max_pool_2d(network, 2)

# Шаг шестой - полносвязная сеть с 1024 нейронами
network = fully_connected(network, 1024, activation='relu')

# Шаг седьмой - выбрасываем некоторые данные из входных, что бы избежать переобучения
# второй аргумент - вероятность сохранения элемента
network = dropout(network, 0.5)

# Шаг восьмой - полносвязная сеть с двумя выходами, что бы сделать окончательное предсказание
network = fully_connected(network, len(EMOTIONS), activation='softmax')

# Определим, как мы будем обучать нейронную сеть - слой регрессии
# определяем оптимизатор градиентного спуска, которы будет применяться для минимизации функции потерь
# так же определяем саму функцию потерь и скорость обучения оптимизатора(град. спуска)
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

# Оборачиваем сеть в модель(объект)
model = tflearn.DNN(network, tensorboard_verbose=0,
                    tensorboard_dir='tflear_logs/', checkpoint_path='checkpoints/emotion-recognizer.tfl.ckpt')

print('Start training...')
# Тренировка
model.fit(X, Y, n_epoch=20, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=25, snapshot_epoch=True, run_id='emotion-recognizer')

# Сохраним модель после обучения
model.save('model/emotion_recognizer.tfl')
print('Network trained and saved as emotion_recognizer.tfl')
