__author__ = 'Artem Pshenichny'


import dash_core_components as dcc
import dash_html_components as html
import numpy as np

from src.constants import *


class Report:
    """
    Class for generating a reports after finish
    """

    def __init__(self, date, times, predictions):
        self.date = date
        self.times = [x.strftime('%H:%M:%S') for x in times]
        self.predictions = predictions

    @staticmethod
    def create_title():
        title = html.H1(
            children='Emotion Recognition Results',
            style={
                'textAlign': 'center'
            }
        )
        return title

    @staticmethod
    def create_desc():
        desc = html.Div(
            children='Description',
            style={
                'textAlign': 'center'
            }
        )
        return desc

    def create_line_chart(self):
        """
        Creates a line chart to shows each emotion values in time
        :return: chart
        """

        plots = [[] for _ in range(0, len(EMOTIONS))]
        for i in range(0, len(EMOTIONS)):
            for p in self.predictions:
                plots[i].append(p[i])

        chart = dcc.Graph(
            id='line-chart',
            figure={
                'data': [{'x': self.times, 'y': p,
                          'type': 'line', 'name': e} for p, e in zip(plots, EMOTIONS)],
                'layout': {'title': 'Emotions by time {} {}-{}'.format(self.date,
                                                                       self.times[0],
                                                                       self.times[1])}
            }
        )
        return chart

    def create_bar_chart(self):
        """
        Creates a bar chart to shows the peaks count of emotions values
        :return: chart
        """

        emotions_count = {x: 0 for x in EMOTIONS}
        for p in self.predictions:
            i = np.argmax(p)
            emotions_count[EMOTIONS[i]] += 1

        chart = dcc.Graph(
            id='bar-chart',
            figure={
                'data': [{'x': [e for e in EMOTIONS], 'y': [emotions_count[e] for e in EMOTIONS],
                          'type': 'bar', 'name': 'Peaks'}],
                'layout': {'title': 'Emotions picks by time {} {}-{}'.format(self.date,
                                                                             self.times[0],
                                                                             self.times[1])}
            }
        )
        return chart

    def create_pie_chart(self):
        """
        Creates a pie chart to shows the share of emotion from the total
        :return: chart
        """

        emotions_count = {x: 0 for x in EMOTIONS}
        for p in self.predictions:
            i = np.argmax(p)
            emotions_count[EMOTIONS[i]] += 1

        sum_picks = sum(emotions_count.values())
        sizes = [x / sum_picks * 100 for x in emotions_count.values()]

        chart = dcc.Graph(
            id='pie-chart',
            figure={
                'data': [{'legend': [e for e in EMOTIONS], 'values': sizes,
                          'type': 'pie', 'name': 'Share of emotions'}],
                'layout': {'title': 'Emotions share by time {} {}-{}'.format(self.date,
                                                                             self.times[0],
                                                                             self.times[1])}
            }
        )
        return chart

    def create_csv(self):
        pass
