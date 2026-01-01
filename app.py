from flask import Flask, request, render_template
from keras.models import load_model
import pickle
import tensorflow
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load model and scaler
model = load_model("AnnModel.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = np.array([
        int(request.form['GRE_Score']),
        int(request.form['TOEFL_Score']),
        int(request.form['University_Rating']),
        float(request.form['SOP']),
        float(request.form['LOR']),
        float(request.form['CGPA']),
        int(request.form['Research'])
    ]).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0][0]
    prediction = max(0, min(1, prediction))

    return str(round(prediction * 100, 2))

if __name__ == "__main__":
    app.run(debug=True)
