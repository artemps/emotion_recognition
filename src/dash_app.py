import webbrowser

import dash
import dash_html_components as html

from report import Report


def app_run(date, times, predictions):
    reporter = Report(date, times, predictions)

    app = dash.Dash()

    app.layout = html.Div(children=[
        reporter.create_title(),
        reporter.create_desc(),
        reporter.create_line_chart(),
        reporter.create_bar_chart(),
        reporter.create_pie_chart()

    ])

    webbrowser.open('http://127.0.0.1:8050/')
    app.run_server(debug=False)

