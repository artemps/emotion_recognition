import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection

from constants import *


class Report:
    def __init__(self, date, times, predictions):
        self.date = date
        self.times = times
        self.predictions = predictions

    def create_line_chart(self):
        # Выбираем значения каждой эмоции из всех предсказаний
        plots = [[] for _ in range(0, len(EMOTIONS))]
        for i in range(0, len(EMOTIONS)):
            for p in self.predictions:
                plots[i].append(p[0][i])

        fig, ax = plt.subplots()
        for emotion, plot in zip(EMOTIONS, plots):
            ax.plot(self.times, plot, label=emotion)

        ax.set(xlabel='time', ylabel='values',
               title='Emotions by time {} {}-{}'.format(self.date, self.times[0], self.times[1]))
        ax.legend()
        ax.autoscale()

        fn = 'line_chart_{}_{}-{}.png'.format(self.date, self.times[0], self.times[1]).replace(':', '_')
        fig.savefig('{}/{}.png'.format(CHARTS_DIR, fn))
        plt.show()

    def create_time_line(self):
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'magenta']
        emotion_colors = {e: c for e, c in zip(EMOTIONS, colors)}

        x = list(range(0, len(self.times)))
        e, y, c = [], [], []
        for p in self.predictions:
            i = np.argmax(p[0])
            y.append(p[0][i])
            e.append(EMOTIONS[i])
            c.append(emotion_colors[EMOTIONS[i]])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, colors=c, linewidths=3)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.set(xlabel='time', ylabel='values',
               title='Emotions by time {} {}-{}'.format(self.date, self.times[0], self.times[1]))
        ax.autoscale()

        patches = [mpatches.Patch(color=c, label=e) for e, c in emotion_colors.items()]
        plt.legend(handles=patches, loc='upper left')

        fn = 'time_line_chart_{}_{}-{}.png'.format(self.date, self.times[0], self.times[1]).replace(':', '_')
        fig.savefig('{}/{}.png'.format(CHARTS_DIR, fn))
        plt.show()

    def create_bar_chart(self):
        emotions_count = {x: 0 for x in EMOTIONS}
        for p in self.predictions:
            i = np.argmax(p[0])
            emotions_count[EMOTIONS[i]] += 1

        fig, ax = plt.subplots()
        for emotion, count in emotions_count.items():
            ax.bar(emotion, count, 0.35, label=emotion)

        ax.set(xlabel='emotions', ylabel='counts',
               title='Emotions pick by time {} {}-{}'.format(self.date, self.times[0], self.times[1]))
        ax.legend()
        ax.autoscale()

        fn = 'bar_chart_{}_{}-{}.png'.format(self.date, self.times[0], self.times[1]).replace(':', '_')
        fig.savefig('{}/{}.png'.format(CHARTS_DIR, fn))
        plt.show()

    def create_pie_chart(self):
        emotions_count = {x: 0 for x in EMOTIONS}
        for p in self.predictions:
            i = np.argmax(p[0])
            emotions_count[EMOTIONS[i]] += 1

        sum_picks = sum(emotions_count.values())
        sizes = [x/sum_picks*100 for x in emotions_count.values()]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=EMOTIONS, autopct='%1.1f%%', startangle=90)
        ax.legend()
        ax.autoscale()

        fn = 'pie_chart_{}_{}-{}.png'.format(self.date, self.times[0], self.times[1]).replace(':', '_')
        fig.savefig('{}/{}.png'.format(CHARTS_DIR, fn))
        plt.show()

    def create_csv(self):
        pass
