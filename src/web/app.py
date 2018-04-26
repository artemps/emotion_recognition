import requests
from flask import Flask, render_template, request, jsonify, abort, make_response

from recognition import recognize

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load', methods=['POST'])
def upload():
    try:
        url = request.form.get('url')
        res = requests.get(url)
        data = recognize(res.content)
        return jsonify(data)
    except Exception as e:
        print(e)
        abort(400)


@app.errorhandler(400)
def bad_request():
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)


if __name__ == '__main__':
    app.run()
