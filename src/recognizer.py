__author__ = 'Artem Pshenichny'


import os
from datetime import datetime
from time import sleep

import cv2
import numpy as np

from constants import *
from detection import Detector
from train_network import TrainNetwork


class Recognizer:
    """
    Class of recognizer. Recognize emotions on image using trained model
    """

    def __init__(self):
        self.detector = Detector()
        self.model = TrainNetwork().load_trained_model()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print('Webcam is not open.')
            exit()

    def recognize(self, frame):
        """
        Recognize emotions on frame for each face
        :param frame: input frame from webcam
        :return: tuple of list faces and predictions
        """

        detected_faces = self.detector.detect_faces(frame)
        predictions = []

        for face in detected_faces:
            face = face.reshape([-1, IMG_SIZE, IMG_SIZE, 1])
            prediction = self.model.predict(face)
            print(EMOTIONS[np.argmax(prediction[0])])
            predictions.append(prediction[0])

        return detected_faces, predictions

    def run(self):
        """
        Gets frames from webcam and push it to recognizer
        Stops by CTRL+C
        :return:
        """
        predictions = []
        times = []
        try:
            while self.cap.isOpened():
                sleep(0.5)
                _, frame = self.cap.read()

                faces, preds = self.recognize(frame)

                for f, p in zip(faces, preds):
                    predictions.append(p)
                    times.append(datetime.now().time())
                    Recognizer._save_recognized_image(frame, p, datetime.now())      

        except KeyboardInterrupt:
            self.cap.release()
            return times, predictions

    @staticmethod
    def _save_recognized_image(face, prediction, dt):
        """
        Saves image from webcam with face(user data)
        :param face: face image
        :param prediction: emotion prediction
        :param dt: datetime
        :return:
        """

        if not os.path.exists(os.path.join(os.getcwd(), USER_DATA_DIR)):
            os.mkdir(USER_DATA_DIR)

        emotion = EMOTIONS[np.argmax(prediction)]
        dt = dt.strftime('%Y_%m_%d_%H_%M_%S')
        cv2.imwrite('{}/{}_{}.jpg'.format(USER_DATA_DIR, emotion, dt), face)
