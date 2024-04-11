import flask
from flask import request
from flask_cors import CORS

app = flask.Flask(__name__)

CORS(app)

# main index page (root route)
@app.route("/")
def home():
    return "<h1>Salary Prediction API</h1><p>BAIS:3300 - Digital Product Development</p><p>Mike Colbert</p>"


# predict route
@app.route("/predict", methods=["GET"])
def predict():
    return "<h1>Prediction route is working... </h1>"


if __name__ == '__main__':
    app.run(debug=True)