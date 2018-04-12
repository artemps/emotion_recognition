__author__ = 'Artem Pshenichny'

import cv2
import glob
import numpy as np
import pandas as pd

from detection import Detector
from constants import *


class PrepareData:
    """
    Class for prepare dataset
    """

    def __init__(self):
        self.images = []
        self.emotions = []

    def create_dataset_from_csv(self):
        """
        Create dataset from fer2013 dataset csv file
        :return: tuple of list dataset images, dataset labels
        """

        print('\nCreating dataset...')
        data = pd.read_csv(CSV_FILE_NAME)
        total = data.shape[0]
        for i, row in data.iterrows():
            image = PrepareData._data_to_image(row['pixels'])
            emotion = PrepareData._emotion_to_vec(row['emotion'])
            self.images.append(image)
            self.emotions.append(emotion)
            print('Progress: {}/{} {:.2f}%'.format(i, total, i * 100.0 / total))

        return self.images, self.emotions

    def add_extra_images(self):
        """
        If --extra flag exists, adds images from extra-images dir to dataset
        :return: tuple of list dataset images, dataset labels
        """

        detector = Detector()
        print('\nAdding extra images...')
        for emotion_index, emotion in enumerate(EMOTIONS):
            print('\nEmotion', emotion)
            files = glob.glob('{}\\{}\\*'.format(EXTRA_DIR, emotion))
            total = len(files)
            for i, f in enumerate(files):
                image = cv2.imread(f)
                detected_faces = detector.detect_faces(image)
                for face in detected_faces:
                    self.images.append(face)
                    self.emotions.append(PrepareData._emotion_to_vec(emotion_index))
                print('Progress: {}/{} {:.2f}%'.format(i, total, i * 100.0 / total))

        return self.images, self.emotions

    @staticmethod
    def _data_to_image(data):
        """
        Private method. Convert csv row of pixels to image
        :param data: row of pixels from csv file
        :return:
        """

        data_image = [int(pixel) for pixel in data.split(' ')]
        image = np.asarray(data_image).reshape(IMG_SIZE, IMG_SIZE)
        image = cv2.resize(image.astype('uint8'), (IMG_SIZE, IMG_SIZE))
        image = image.astype('float32') / 255.
        return image

    @staticmethod
    def _emotion_to_vec(x):
        """
        Private method. Convert num of emotion to emotions labels array([0., 0., 0. ,1., 0., 0., 0.,])
        :param x: num of emotion
        :return: array
        """

        d = np.zeros(len(EMOTIONS))
        d[x] = 1.0
        return d
