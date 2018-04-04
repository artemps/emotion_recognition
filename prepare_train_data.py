import os
import glob
import cv2
from shutil import copy

import detection


def prepare_CK():
    """
    Датасет CK+
    Подготовка данных для тренировки - разбивает изображения по папкам эмоций в папку sorted-sets.
    Данные находятся в папках images - изображения эмоций, где верхние папки - это один человек, 
    папки внутри - это эмоции человека, а файлы идут от первого(нейтрального) кадра к конечному(эмоциональному)
    и emotions - текстовый файл, описывающий в себе номер эмоции
    :return: количество отсортированных изображений
    """

    # Эмоции из датасета CK+, данные по которым будут подготавливаться
    # emotions = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    folders = glob.glob('emotions\\*')

    count = 0
    for x in folders:
        folder_code = "%s" % x[-4:]

        first_neut = True  # Сохраняем только один нейтральный файл для всех сессий

        for sessions in glob.glob('%s\\*' % x):
            for file in glob.glob('%s\\*' % sessions):
                current_session = file[14:17]  # НЕ ТРОГАТЬ ИНДЕКСЫ И НЕ МЕНЯТЬ НАЗВАНИЯ ПАПОК
                file = open(file, 'r')

                # информация в файле - это номер эмоции человека, по которому мы можем узнать саму эмоцию
                emotion = int(float(file.readline()))

                if first_neut:
                    # Берем нейтральный(только один раз)
                    file_neutral = glob.glob('images\\%s\\%s\\*' % (folder_code, current_session))[0]
                    dest_neutral = 'sorted-sets\\neutral\\%s' % file_neutral[16:]  # НЕ ТРОГАТЬ ИНДЕКСЫ И НЕ МЕНЯТЬ НАЗВАНИЯ ПАПОК
                    copy(file_neutral, dest_neutral)

                    first_neut = False
                    count += 1

                #  и последний(эмоциональный) файлы
                file_emotion = glob.glob('images\\%s\\%s\\*' % (folder_code, current_session))[-1]
                dest_emotion = 'sorted-sets\\%s\\%s' % (emotions[emotion], file_emotion[16:])  # НЕ ТРОГАТЬ ИНДЕКСЫ И НЕ МЕНЯТЬ НАЗВАНИЯ ПАПОК
                copy(file_emotion, dest_emotion)

                count += 1

    return count


def preparing_data():
    """
    Датасет CK+ и директория с дополнительными изображениями
    Подготовка данных для тренировки - выделяет на каждом изображении лицо, конвертирует изображение в градации серого
    и отцентровывает его.
    Конечные изображения разбиваются по папка эмоций в папке datasets
    :param extra: Флаг, нужно ли использовать дополнительные изображения
    :return: количество обработанных изображений
    """

    # Эмоции(contempt эмоция из датасета CK+ пропускается)
    # emotions = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    emotions = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    count = 0
    for emotion in emotions:
        files = glob.glob('sorted-sets\\%s\\*' % emotion)
        extra_data_files = glob.glob('extra-images\\%s\\*' % emotion)
        files = files + extra_data_files

        i = 0
        for f in files:
            try:
                image = cv2.imread(f)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                aligned_images, _ = detection.detect_face(gray_image)
            except Exception as e:
                print(e)
                os.remove(f)
                continue

            # Удаляем файлы, на которых не были найдены лица
            if aligned_images:
                for img in aligned_images:
                    cv2.imwrite('datasets\\%s\\%s.jpg' % (emotion, i), img)
                    i += 1
            else:
                os.remove(f)

        count += i

    return count
