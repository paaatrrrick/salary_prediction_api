import flask
from flask import request
from flask_cors import CORS
import joblib
model = joblib.load('salary_predict_model.ml')
app = flask.Flask(__name__)

CORS(app)

# main index page (root route)
@app.route("/")
def home():
    return "<h1>Salary Prediction API</h1><p>BAIS:3300 - Digital Product Development</p><p>Mike Colbert</p>"


# predict route
@app.route("/predict", methods=["post"])
def predict():
    json = request.json
    arr = [json["age"],
           json["gender"],
           json["country"],
           json["highest_deg"],
           json["coding_exp"],
           json["title"],
           json["company_size"]]
    predctions = model.predict([arr]).tolist()
    return {"predicted_salary" : predctions[0]}



if __name__ == '__main__':
    app.run(debug=True)