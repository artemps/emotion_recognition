import os
import shutil
from time import sleep

import cv2
import sys
import tflearn
import numpy as np
from datetime import datetime

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import detection
from create_charts import create_line_chart, create_time_line, create_bar_chart, create_pie_chart
from generate_csv import generate_csv_data


# нейтральный, злой, отвращение, страх, счастливый, грустный, удивленный
# EMOTIONS = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def recognize(image, model):
    """
    Распознает эмоции на входном изображении
    :param image: входное изображение
    :return: первый элемент - все выделеные лица, второй элемент - вероятность нахождения каждой эмоции
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Пропускаем через детектор
    aligned_images, face_rects = detection.detect_face(gray_image)
    aligned_images = [x / 255. for x in aligned_images]
    aligned_images = np.reshape(aligned_images, [-1, 64, 64, 1])

    predictions = []
    for face in aligned_images:
        prediction = model.predict([face])
        predictions.append(prediction[0])

    return face_rects, predictions


def define_model():
    # Определяем ту сеть, которой будем распознавать эмоции
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.0)
    img_aug.add_random_blur(sigma_max=3.0)

    network = input_data(shape=[None, 64, 64, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, len(EMOTIONS), activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='checkpoints/emotion-classifier.tfl.ckpt')
    model.load('model/emotion_recognizer.tfl')

    return model


def main(model):
    if os.path.exists(os.path.join(os.getcwd(), 'recognized_emotions')):
        shutil.rmtree('recognized_emotions')
        os.mkdir('recognized_emotions')
    else:
        os.mkdir('recognized_emotions')

    if not os.path.exists(os.path.join(os.getcwd(), 'charts')):
        os.mkdir('charts')

    print('Starting...')

    cap = cv2.VideoCapture(0)
    i = 1
    emotions_predictions = []
    times = []

    try:
        while cap.isOpened():
            sleep(0.5)
            _, frame = cap.read()
            rectangles, predictions = recognize(frame, model)

            # Отрисовка квадратов лиц и подписей
            for rect, pred in zip(rectangles, predictions):
                cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                cv2.imwrite('recognized_emotions/images_{}.jpg'.format(i), frame)

                emotions_predictions.append(predictions)
                times.append(datetime.now().time())
                i += 1

    except KeyboardInterrupt:
        cap.release()

    return times, emotions_predictions

if __name__ == '__main__':
    model = define_model()
    t, ep = main(model)

    if not t or not ep:
        print('Faces not found')
        sys.exit()

    # Create charts
    date = datetime.now().date().strftime('%Y-%m-%d')
    t = [x.strftime('%H:%M:%S') for x in t]

    create_line_chart(date, t, EMOTIONS, ep)
    create_time_line(date, t, EMOTIONS, ep)
    create_bar_chart(date, t, EMOTIONS, ep)
    create_pie_chart(date, t, EMOTIONS, ep)

    # Create csv
    generate_csv_data(date, t, EMOTIONS, ep)
