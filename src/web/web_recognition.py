import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))  # hack for importing from src

import cv2
import numpy as np

from constants import *
from recognizer import Recognizer


recognizer = Recognizer(with_webcam=False)


def web_recognize(data):

    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face_rects = recognizer.detector.get_faces_rects(image)
    _, predictions = recognizer.recognize(image)

    resp = {'Faces': []}

    for i, face in enumerate(face_rects):
        _obj = {'Face Rectangles': {'x': int(face[0]),
                                    'y': int(face[1]),
                                    'width': int(face[2]),
                                    'height': int(face[3])},
                'Emotions': {e: round(float(p), 3)
                             for e, p in zip(EMOTIONS, predictions[i])}}

        resp['Faces'].append(_obj)

    return resp