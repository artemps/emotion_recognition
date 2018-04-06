import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from detection import Detector
from constants import *


class PrepareData:
    @staticmethod
    def create_dataset_from_csv():
        print('Start creating dataset...')

        PrepareData._make_dirs()

        data = pd.read_csv(CSV_FILE_NAME)
        i = 1
        total = data.shape[0]
        for index, row in data.iterrows():
            emotion = row['emotion']
            image = PrepareData._data_to_image(row['pixels'])

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector = Detector()
            detected_faces = detector.detect_faces(gray_image)
            aligned_faces, _ = detector.align_faces(gray_image, detected_faces)
            for x in aligned_faces:
                x.save('{}/{}/{}.png'.format(DATA_DIR, EMOTIONS[emotion], i), 'PNG')
                index += 1
                print('Progress: {}/{} {:.2f}%'.format(index, total, index * 100.0 / total))

    @staticmethod
    def _make_dirs():
        if not os.path.exists(os.path.join(os.getcwd(), DATA_DIR)):
            os.mkdir(os.path.join(os.getcwd(), DATA_DIR))
            for emotion in EMOTIONS:
                os.mkdir(os.path.join(os.getcwd(), os.path.join(DATA_DIR, emotion)))

    @staticmethod
    def _data_to_image(data):
        data_image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((IMG_SIZE, IMG_SIZE))
        image = Image.fromarray(data_image).convert('RGB')
        return image
