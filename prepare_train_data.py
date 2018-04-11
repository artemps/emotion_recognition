import os
import cv2
import glob
import numpy as np
from PIL import Image
import pandas as pd

from detection import Detector
from constants import *


class PrepareData:
    @staticmethod
    def create_dataset_from_csv():
        if os.path.exists(os.path.join(os.getcwd(), DATA_DIR)):
            print('\nDataset dir already exists.')
            return

        print('\nCreating dataset...')
        PrepareData._make_dirs()

        data = pd.read_csv(CSV_FILE_NAME)
        total = data.shape[0]
        detector = Detector()
        for i, row in data.iterrows():
            emotion = row['emotion']
            image = PrepareData._data_to_image(row['pixels'])
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = detector.detect_faces(gray_image)
            # aligned_faces = detector.align_faces(gray_image, detected_faces)
            # for x in aligned_faces:
            if len(detected_faces) > 0:
                cv2.imwrite('{}/{}/{}.png'.format(DATA_DIR, EMOTIONS[emotion], i), gray_image)
                print('Progress: {}/{} {:.2f}%'.format(i, total, i * 100.0 / total))

    @staticmethod
    def add_extra_images():
        if not os.path.exists(os.path.join(os.getcwd(), EXTRA_DIR)):
            print('\nExtra images dir does not exist')
            return

        detector = Detector()

        print('\nAdding extra images...')
        for emotion in EMOTIONS:
            print('\nEmotion', emotion)
            files = glob.glob('{}\\{}\\*'.format(EXTRA_DIR, emotion))
            i = 0
            total = len(files)
            for f in files:
                image = cv2.imread(f)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected_faces = detector.detect_faces(gray_image)
                aligned_faces = detector.align_faces(gray_image, detected_faces)
                for x in aligned_faces:
                    cv2.imwrite('{}/{}/extra_{}.png'.format(DATA_DIR, emotion, i), x)
                    i += 1
                    print('Progress: {}/{} {:.2f}%'.format(i, total, i * 100.0 / total))

    @staticmethod
    def _data_to_image(data):
        data_image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((IMG_SIZE, IMG_SIZE))
        data_image = Image.fromarray(data_image).convert('RGB')
        data_image = np.array(data_image)[:, :, ::-1].copy()
        return data_image

    @staticmethod
    def _make_dirs():
        print('Creating dirs...\n')
        os.mkdir(os.path.join(os.getcwd(), DATA_DIR))
        for emotion in EMOTIONS:
            os.mkdir(os.path.join(os.getcwd(), os.path.join(DATA_DIR, emotion)))