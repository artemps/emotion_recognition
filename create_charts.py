import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def create_line_chart(date, times, predictions):
    anger = [x[0][0] for x in predictions]
    disgust = [x[0][1] for x in predictions]
    fear = [x[0][2] for x in predictions]
    happy = [x[0][3] for x in predictions]
    sadness = [x[0][4] for x in predictions]
    surprise = [x[0][5] for x in predictions]

    fig, ax = plt.subplots()
    ax.plot(times, anger, label='Anger')
    ax.plot(times, disgust, label='Disgust')
    ax.plot(times, fear, label='Fear')
    ax.plot(times, happy, label='Happy')
    ax.plot(times, sadness, label='Sadness')
    ax.plot(times, surprise, label='Surprise')

    ax.set(xlabel='time', ylabel='emotions',
           title='Emotions by time {} {}-{}'.format(date, times[0], times[1]))
    ax.legend()

    fn = 'line_chart_{}_{}-{}.png'.format(date, times[0], times[1]).replace(':', '_')
    fig.savefig('charts/{}.png'.format(fn))
    plt.show()


def create_time_line(date, times, emotions, predictions):
    x = np.arange(10)
    y = [1] * 10

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = ['r', 'g', 'b']

    lc = LineCollection(segments, colors=colors, linewidths=5)
    fig, ax = plt.subplots()
    ax.add_collection(lc)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()


def create_bar_chart(date, emotions, predictions):
    men_means = (20, 35, 30, 35, 27)

    ind = [1, 2, 3, 4, 5]  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, men_means, width,
                    color='SkyBlue', label='Men')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind)
    ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    ax.legend()

    plt.show()


def create_pie_chart(date, emotions, predictions):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()