__author__ = 'Artem Pshenichny'


import os
from time import sleep
from datetime import datetime

import cv2
import numpy as np

from train_network import TrainNetwork
from detection import Detector
from report import Report
from constants import *


class Recognizer:
    """
    Class of recognizer. Recognize emotions on image using trained model
    """

    def __init__(self):
        self.detector = Detector()
        self.cap = cv2.VideoCapture(0)

        t = TrainNetwork()
        self.model = t.load_trained_model()

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

        Recognizer._make_dirs()

        predictions = []
        times = []
        try:
            while self.cap.isOpened():
                sleep(0.5)
                _, frame = self.cap.read()

                faces, preds = self.recognize(frame)

                for f, p in zip(faces, predictions):
                    predictions.append(p)
                    Recognizer._save_recognized_image(f*255., p, datetime.now())
                    times.append(datetime.now().time())

        except KeyboardInterrupt:
            self.cap.release()

    @staticmethod
    def create_report(times, predictions):
        """
        Creates finish report
        :param times: arrays of times when whas frames from webcam
        :param predictions: arrays of predictions
        :return:
        """

        date = datetime.now().date().strftime('%Y-%m-%d')
        times = [x.strftime('%H:%M:%S') for x in times]

        report = Report(date, times, predictions)

        report.create_line_chart()
        report.create_time_line()
        report.create_bar_chart()
        report.create_pie_chart()
        report.create_csv()

    @staticmethod
    def _save_recognized_image(face, prediction, dt):
        """
        Saves image from webcam with face(user data)
        :param face: face image
        :param prediction: emotion prediction
        :param dt: datetime
        :return:
        """

        emotion = EMOTIONS[np.argmax(prediction)]
        dt = dt.strftime('%Y_%m_%d_%H_%M_%S')
        cv2.imwrite('{}/{}_{}.jpg'.format(USER_DATA_DIR, emotion, dt), face)

    @staticmethod
    def _make_dirs():
        """
        Makes dirs for user images(user data)
        :return:
        """

        if not os.path.exists(os.path.join(os.getcwd(), USER_DATA_DIR)):
            os.mkdir(USER_DATA_DIR)
        if not os.path.exists(os.path.join(os.getcwd(), CHARTS_DIR)):
            os.mkdir(CHARTS_DIR)
