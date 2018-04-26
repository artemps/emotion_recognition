""" Entry point of application """
__author__ = 'Artem Pshenichny'


import sys
from datetime import datetime

from constants import *
from dash_app import app_run


def show_usage():
    print('Usage: python emotion_recognition.py')
    print('\t emotion_recognition.py train \t Saves dataset and trains and saves model')
    print('\t emotion_recognition.py train --extra \t If you want to add extra images to dataset')
    print('\t emotion_recognition.py start \t Starts app')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        show_usage()
        exit()

    # Train network
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

    # Start emotion recognition
    elif sys.argv[1] == 'start':
        from recognizer import Recognizer

        recognizer = Recognizer()
        times, predictions = recognizer.run()
        date = datetime.now().date().strftime('%Y-%m-%d')

        # Start dash app to shows charts
        app_run(date, times, predictions)

    else:
        show_usage()
