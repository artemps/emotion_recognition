import sys

from prepare_train_data import PrepareData
from train_network import TrainNetwork


def show_usage():
    print('Usage: python emotion_recognition.py')
    print('\t emotion_recognition.py train \t Saves dataset and trains and saves model')
    print('\t emotion_recognition.py start \t Starts app')


def main():
    pass

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        show_usage()
        exit()

    if sys.argv[1] == 'train':
        prep = PrepareData()
        prep.create_dataset_from_csv()

        trainer = TrainNetwork()
        model = trainer.define_network()
        trainer.start_fit(model)
    elif sys.argv[1] == 'start':
        pass
    else:
        show_usage()