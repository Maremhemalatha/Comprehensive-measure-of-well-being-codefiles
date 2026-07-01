from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)


print("Current Working Directory:", os.getcwd())
print("App Root Path:", app.root_path)


# Load the trained model
model = pickle.load(open("HDI.pkl", "rb"))

# Load dataset
df = pd.read_csv("../Dataset/HDI.csv")

# Get all country names
countries = sorted(df["Country"].dropna().unique())

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("indexnew.html", countries=countries)

@app.route("/predict_result", methods=["POST"])
def predict_result():

    life = float(request.form["life"])
    expected = float(request.form["expected"])
    mean = float(request.form["mean"])
    gni = float(request.form["gni"])

    features = np.array([[life, expected, mean, gni]])

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 3)
    )

if __name__ == "__main__":
    app.run(debug=True)