import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection


def create_line_chart(date, times, emotions, predictions):

    # Выбираем значения каждой эмоции из всех предсказаний
    plots = [[], [], [], [], [], []]
    for i in range(0, len(emotions)):
        for p in predictions:
            plots[i].append(p[0][i])

    fig, ax = plt.subplots()
    for emotion, plot in zip(emotions, plots):
        ax.plot(times, plot, label=emotion)

    ax.set(xlabel='time', ylabel='values',
           title='Emotions by time {} {}-{}'.format(date, times[0], times[1]))
    ax.legend()
    ax.autoscale()

    fn = 'line_chart_{}_{}-{}.png'.format(date, times[0], times[1]).replace(':', '_')
    fig.savefig('charts/{}.png'.format(fn))
    plt.show()


def create_time_line(date, times, emotions, predictions):
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    emotion_colors = {e: c for e, c in zip(emotions, colors)}

    x = list(range(0, len(times)))
    e, y, c = [], [], []
    for p in predictions:
        i = np.argmax(p[0])
        y.append(p[0][i])
        e.append(emotions[i])
        c.append(emotion_colors[emotions[i]])

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, colors=c, linewidths=3)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.set(xlabel='time', ylabel='values',
           title='Emotions by time {} {}-{}'.format(date, times[0], times[1]))
    ax.autoscale()

    patches = [mpatches.Patch(color=c, label=e) for e, c in emotion_colors.items()]
    plt.legend(handles=patches, loc='upper left')

    fn = 'time_line_chart_{}_{}-{}.png'.format(date, times[0], times[1]).replace(':', '_')
    fig.savefig('charts/{}.png'.format(fn))
    plt.show()


def create_bar_chart(date, times, emotions, predictions):

    # Считаем кол-во пиков(самых больших значений из всех) каждой эмоции за все предсказаний
    emotions_count = {x: 0 for x in emotions}
    for p in predictions:
        i = np.argmax(p[0])
        emotions_count[emotions[i]] += 1

    fig, ax = plt.subplots()
    for emotion, count in emotions_count.items():
        ax.bar(emotion, count, 0.35, label=emotion)

    ax.set(xlabel='emotions', ylabel='counts',
           title='Emotions pick by time {} {}-{}'.format(date, times[0], times[1]))
    ax.legend()
    ax.autoscale()

    fn = 'bar_chart_{}_{}-{}.png'.format(date, times[0], times[1]).replace(':', '_')
    fig.savefig('charts/{}.png'.format(fn))
    plt.show()


def create_pie_chart(date, times, emotions, predictions):

    # Считаем кол-во пиков(самых больших значений из всех) каждой эмоции за все предсказаний
    emotions_count = {x: 0 for x in emotions}
    for p in predictions:
        i = np.argmax(p[0])
        emotions_count[emotions[i]] += 1

    sum_picks = sum(emotions_count.values())
    sizes = [x/sum_picks*100 for x in emotions_count.values()]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=emotions, autopct='%1.1f%%', startangle=90)
    ax.legend()
    ax.autoscale()

    fn = 'pie_chart_{}_{}-{}.png'.format(date, times[0], times[1]).replace(':', '_')
    fig.savefig('charts/{}.png'.format(fn))
    plt.show()

