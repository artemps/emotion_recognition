import dlib
import openface

from constants import *


class Detector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_aligner = openface.AlignDlib(PREDICTOR_MODEL)

    def detect_faces(self, image):
        detected_faces = self.face_detector(image, 1)
        return detected_faces

    def align_faces(self, image, detected_faces):
        aligned_faces = []
        face_rects = []
        for face in detected_faces:
            aligned_face = self.face_aligner.align(48, image, face,
                                                   landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            aligned_faces.append(aligned_face)
            face_rects.append((face.left(), face.top(), face.right(), face.bottom()))

        return aligned_faces, face_rects
