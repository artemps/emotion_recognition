import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))  # hack for importing from src

import cv2
import numpy as np

from constants import *
from recognizer import Recognizer

recognizer = Recognizer()


def recognize(data):

    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face_rects = recognizer.detector.get_faces_rects(image)
    _, predictions = recognizer.recognize(image)

    mes = {'Faces Count': len(face_rects),
           'Faces': {}}

    for i, face in enumerate(face_rects):
        coords = {'Coordinates': {'x': face[0],
                                  'y': face[1],
                                  'width': face[2],
                                  'height': face[3]}}

        emotions = {e: round(p, 3) for e, p in zip(EMOTIONS, predictions[i])}
        mes['Faces'][i] = [coords, emotions]

    return mes