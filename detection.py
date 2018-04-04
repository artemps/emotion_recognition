import dlib
import openface

face_detector = dlib.get_frontal_face_detector()

predictor_model = 'shape_predictor_68_face_landmarks.dat'
face_aligner = openface.AlignDlib(predictor_model)


def detect_face(image):
    """
    Выделить и отцентровать лица на входном изображении    
    :param image: входное изображение 
    :return: список изображение отцентрованных лиц, список кортежей с точками квадратов лиц
    """

    detected_faces = face_detector(image, 1)

    aligned_faces = []
    face_rects = []

    for face_rect in detected_faces:
        aligned_face = face_aligner.align(64, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        aligned_faces.append(aligned_face)
        face_rects.append((face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    return aligned_faces, face_rects
