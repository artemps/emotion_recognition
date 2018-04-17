""" Entry point of application """
__author__ = 'Artem Pshenichny'


import sys
from constants import *


def show_usage():
    print('Usage: python emotion_recognition.py')
    print('\t emotion_recognition.py train \t Saves dataset and trains and saves model')
    print('\t emotion_recognition.py train --extra \t If you want to add extra images to dataset')
    print('\t emotion_recognition.py start \t Starts app')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        show_usage()
        exit()

    if sys.argv[1] == 'train':
        from prepare_train_data import PrepareData
        from train_network import TrainNetwork

        prep = PrepareData()
        images, emotions, count = prep.create_dataset_from_csv()
        if len(sys.argv) > 2 and sys.argv[2] == '--extra':
            prep.add_extra_images()

        for i, emotion in enumerate(EMOTIONS):
            print('{}: {}'.format(emotion, count.count(i)))

        trainer = TrainNetwork(images, emotions)
        trainer.start_train()

    elif sys.argv[1] == 'start':
        from recognizer import Recognizer

        recognizer = Recognizer()
        recognizer.run()

    else:
        show_usage()
