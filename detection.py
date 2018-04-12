__author__ = 'Artem Pshenichny'


import cv2

from constants import *


class Detector:
    """
    Detector class for detection faces on dataset images
    """

    def __init__(self):
        self.cascade_classifier = cv2.CascadeClassifier()

    def detect_faces(self, image):
        """
        Detects faces on input image
        :param image: imput image
        :return: list of cropped images(with faces)
        """

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
