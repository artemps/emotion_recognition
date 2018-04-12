__author__ = 'Artem Pshenichny'


import cv2
import numpy as np

from constants import *


class Detector:
    """
    Detector class for detection faces on dataset images
    """

    def __init__(self):
        self.cascade_classifier = cv2.CascadeClassifier()

    def detect_faces(self, image, fer2013_image=False):
        """
        Detects faces on input image
        :param image: input image
        :param fer2013_image: flag, if image from fer1013 dataset than draws borden near image
        :return: list of cropped images(with faces)
        """

        if fer2013_image:
            gray_border = np.zeros((150, 150), np.uint8)
            gray_border[:, :] = 200
            gray_border[(150 // 2) - (IMG_SIZE // 2):(150 // 2) + (IMG_SIZE // 2),
                        (150 // 2) - (IMG_SIZE // 2):(150 // 2) + (IMG_SIZE // 2)] = image
            image = gray_border

        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return []

        faces_images = []
        for face in faces:
            face_image = image[face[1]:face[1] + face[2], face[0]:face[0] + face[3]]
            face_image = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))
            face_image = face_image.astype('float32') / 255.
            faces_images.append(face_image)

        return faces_images
