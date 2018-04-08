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

from constants import *


class TrainNetwork:
    def __init__(self):
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.network = None

    def define_network(self):
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

        self.network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1],
                                  data_preprocessing=img_prep,
                                  data_augmentation=img_aug)

        self.network = conv_2d(self.network, 64, 3, activation='relu')
        self.network = max_pool_2d(self.network, 2)
        self.network = conv_2d(self.network, 128, 3, activation='relu')
        self.network = conv_2d(self.network, 128, 3, activation='relu')
        self.network = max_pool_2d(self.network, 2)
        self.network = fully_connected(self.network, 1024, activation='relu')
        self.network = dropout(self.network, 0.5)
        self.network = fully_connected(self.network, len(EMOTIONS), activation='softmax')
        self.network = regression(self.network, optimizer='adam',
                                  loss='categorical_crossentropy', learning_rate=0.001)
        model = tflearn.DNN(self.network, tensorboard_verbose=0,
                            checkpoint_path='checkpoints')

        return model

    def start_fit(self, model):
        print('\nEmotions:', EMOTIONS)
        print('Start training...')

        (x, y), (x_test, y_test) = self.make_sets()
        x, y = shuffle(x, y)
        y = to_categorical(y, len(EMOTIONS))
        y_test = to_categorical(y_test, len(EMOTIONS))

        model.fit(x, y, n_epoch=100, shuffle=True, validation_set=(x_test, y_test),
                  show_metric=True, batch_size=50, snapshot_epoch=True, run_id='emotion-recognizer')

        model.save('model/emotion_recognizer.tfl')
        print('\nNetwork trained and saved as emotion_recognizer.tfl')

    def make_sets(self):
        for emotion in EMOTIONS:
            train, test = TrainNetwork.get_files(emotion)
            for item in train:
                image = cv2.imread(item)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
                self.train_images.append(gray_image)
                self.train_labels.append(EMOTIONS.index(emotion))

            for item in test:
                image = cv2.imread(item)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
                self.test_images.append(gray_image)
                self.test_labels.append(EMOTIONS.index(emotion))

        train_data = np.reshape(self.train_images, [-1, IMG_SIZE, IMG_SIZE, 1])
        test_data = np.reshape(self.test_images, [-1, IMG_SIZE, IMG_SIZE, 1])

        return (train_data, self.train_labels), \
               (test_data, self.test_labels)

    @staticmethod
    def get_files(emotion):
        files = glob.glob('{}\\{}\\*'.format(DATA_DIR, emotion))
        random.shuffle(files)
        train = files[:int(len(files) * 0.8)]  # Берем первые 80% файлов эмоции для тренировки
        test = files[-int(len(files) * 0.2):]  # и последние 20% для тестирования
        return train, test
