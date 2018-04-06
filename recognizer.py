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
    detector = Detector()
    cap = cv2.VideoCapture(0)

    @staticmethod
    def recognize(image, model):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Пропускаем через детектор
        detected_faces = Recognizer.detector.detect_faces(gray_image)
        aligned_faces = Recognizer.detector.align_faces(gray_image, detected_faces)
        aligned_faces = [x / 255. for x in aligned_faces]
        aligned_faces = np.reshape(aligned_faces, [-1, IMG_SIZE, IMG_SIZE, 1])

        predictions = []
        for face in aligned_faces:
            prediction = model.predict([face])
            predictions.append(prediction)

        return aligned_faces, predictions

    @staticmethod
    def run():
        Recognizer._make_dirs()
        model = TrainNetwork().define_network()

        predictions = []
        times = []
        try:
            while Recognizer.cap.isOpened():
                sleep(0.5)
                _, frame = Recognizer.cap.read()
                faces, pred = Recognizer.recognize(frame, model)

                i = 0
                for f, p in zip(faces, predictions):
                    predictions.append(p)
                    times.append(datetime.now().time())
                    Recognizer._save_recognized_image(f, p, datetime.now())
                    i += 1

        except KeyboardInterrupt:
            Recognizer.cap.release()
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
        emotion = EMOTIONS[np.argmax(prediction[0])]
        dt = dt.strftime('%Y_%m_%d_%H_%M_%S')
        cv2.imwrite('{}/{}_{}.jpg'.format(USER_DATA_DIR, emotion, dt), face)

    @staticmethod
    def _make_dirs():
        if not os.path.exists(os.path.join(os.getcwd(), USER_DATA_DIR)):
            os.mkdir(USER_DATA_DIR)
        if not os.path.exists(os.path.join(os.getcwd(), CHARTS_DIR)):
            os.mkdir(CHARTS_DIR)
