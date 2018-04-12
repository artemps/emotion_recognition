__author__ = 'Artem Pshenichny'


import numpy as np
import os
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing

from constants import *


class TrainNetwork:
    """
    Trainer class for train CNN
    """

    def __init__(self, images=None, emotions=None):
        if not images or not emotions:
            self.train_data = None
            self.val_data = None
        else:
            self.train_data, self.val_data = TrainNetwork.split_data(np.asarray(images), np.asarray(emotions), 0.2)

        self.network = None
        self.model = None

    def define_network(self):
        """
        Defines CNN architecture
        :return: CNN model
        """

        # For data normalization
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

        # For creating extra data(increase dataset). Flipped, Rotated, Blurred and etc. images
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=25.0)
        img_aug.add_random_blur(sigma_max=3.0)

        self.network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1],
                                  data_augmentation=img_aug,
                                  data_preprocessing=img_prep)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(self.network, len(EMOTIONS), activation='softmax')
        self.network = regression(self.network, optimizer='adam', loss='categorical_crossentropy')
        self.model = tflearn.DNN(self.network, checkpoint_path=os.path.join(CHECKPOINTS_PATH + '/emotion_recognition'),
                                 max_checkpoints=1, tensorboard_verbose=0)

        return self.model

    def start_train(self):
        """
        Starts train defined model
        :return:
        """

        print('\nEmotions:', EMOTIONS)
        print('Start training...')

        if not self.train_data or not self.val_data:
            print('\nDataset is empty. Exit.')
            exit()

        self.define_network()
        self.model.fit(self.train_data[0], self.train_data[1], validation_set=self.val_data,
                       n_epoch=100, batch_size=64, shuffle=True, show_metric=True,
                       snapshot_epoch=True, snapshot_step=200, run_id='emotion-recognition')

        self.model.save(os.path.join(MODEL_DIR, 'emotion_recognizer'))
        print('\nNetwork trained and saved as emotion_recognizer')

    def load_trained_model(self):
        """
        Loads trained model from save dir
        :return:
        """

        model = self.define_network()
        model.load(os.path.join(MODEL_DIR, 'emotion_recognizer'))
        return model

    @staticmethod
    def split_data(images, emotions, validation_split=0.2):
        """
        Splits input data to train and val
        :param images: array of input images
        :param emotions: array of input labels
        :param validation_split: val data size from train data
        :return: tuple of tuples (train_x, train_y), (val_x, val_y)
        """

        num_samples = len(images)
        num_train_samples = int((1 - validation_split) * num_samples)
        train_x = images[:num_train_samples]
        train_y = emotions[:num_train_samples]
        val_x = images[num_train_samples:]
        val_y = emotions[num_train_samples:]

        train_data = (train_x.reshape([-1, IMG_SIZE, IMG_SIZE, 1]), train_y)
        val_data = (val_x.reshape([-1, IMG_SIZE, IMG_SIZE, 1]), val_y)

        return train_data, val_data
