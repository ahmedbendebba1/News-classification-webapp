from ml_models.model_package_sklearn import ModelPackageSK
from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd

# import en_core_web_md
import en_core_web_sm
import torch
import random
import time
import os

# initialize flask application
app = Flask(__name__)

# Fixing all seeds
def random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python


random_seed(666)

# Define SpaCy model
nlp = en_core_web_sm.load(disable=["tagger", "parser", "ner"])

# Load pre-trained models
# model_svm = ModelPackageSK("/app/sklearn_models/saved_models", "svc_all.joblib", nlp)
model_logreg = ModelPackageSK(
    "app/sklearn_models/saved_models", "logregression_all.joblib", nlp
)

# Load data for random sampling of news
data = pd.read_feather("app/data/dataset_inference")


# Define API
HOST = "0.0.0.0"
PORT = os.environ.get("PORT", 8081)


@app.route("/")
def index():
    return render_template("index.html", prediction={})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
	For rendering results on HTML GUI
	"""
    if request.method == "POST":

        if request.form["submit_button"] == "Generate":
            i = random.sample(list(data.index), 1)[0]
            news_text = data["text"][i]
            return render_template("index.html", prediction={}, news=str(news_text))

        elif request.form["submit_button"] == "TF-IDF & Logistic Regression":
            text_in = request.form["text"]
            if len(text_in) == 0:
                return render_template(
                    "index.html",
                    prediction={"ERROR": "Please enter a headline or generate one"},
                )
            else:
                # Check if text_in is in inference data
                filter1 = data["text"].isin([text_in])
                true_value = (
                    data["category"][filter1].values[0]
                    if filter1.any()
                    else "does not exist in the inference dataset"
                )

                output = model_logreg.topk_predictions(text_in, 3)
                return render_template(
                    "index.html",
                    prediction=output,
                    news=str(text_in),
                    real=str(true_value),
                )
	


        elif request.form["submit_button"] == "Clear":
            return render_template("index.html", prediction={}, news="")

        else:
            pass

    elif request.method == "GET":
        return render_template("index.html", prediction={})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
