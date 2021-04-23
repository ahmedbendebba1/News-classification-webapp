"""
Run : 
Logistic regression: 
python3 baseline.py --model lreg --exp_name logregression_balanced --data dataset_processed_balanced
python3 baseline.py --model lreg --exp_name logregression_all --data dataset_processed
SVC: 
python3 baseline.py --model svc --exp_name svc_balanced --data dataset_processed_balanced
python3 baseline.py --model svc --exp_name svc_all --data dataset_processed
"""
import pandas as pd
import numpy as np
import utils
import argparse
import os
from functools import partial
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split


# Absolute PATH where data folder is
PATH = ".."

# These params have been found using params_search.py
MODELS = {
    # 0.2 for balanced
    "svc": CalibratedClassifierCV(LinearSVC(C=0.5)),
    "lreg": LogisticRegression(C=5),
}


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(
        name, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=" ")
    return parser


class ModelConfig(argparse.Namespace):  # noqa
    """
    Training configuration
    """
    def build_parser(self):
        parser = get_parser("Model training")
        parser.add_argument("--model", required=True)
        parser.add_argument("--exp_name", required=True)
        parser.add_argument("--data", default="dataset_processed_balanced")

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))


# Define config
config = ModelConfig()

# Define logger
logger = utils.get_logger("logs/{}.log".format(config.exp_name))


def main():
    # Loading Dataset
    logger.info("Start Loading saved feather dataset")
    data = pd.read_feather(f"{PATH}/data/{config.data}")
    logger.info("Data loaded.")

    # TF-IDF transformer + simple algorithm
    # The params for the model have been found using params_search.py
    logger.info("Define the pipeline:")
    text_classifier = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_df=0.9,
                    min_df=3,
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("clf", MODELS[config.model]),
        ]
    )
    logger.info(str(text_classifier))

    # Split data : training/testing
    logger.info("Training, testing split.")
    xtrain, xtest, ytrain, ytest = train_test_split(
        data["clean_text"], data.category, random_state=42, test_size=0.1
    )

    # Train model and evaluate
    logger.info("Training the classifier...")
    text_classifier.fit(xtrain, ytrain)

    # Scoring
    logger.info("Training is done. F1-score:")
    prediction = text_classifier.predict(xtest)
    prediction_proba = text_classifier.predict_proba(xtest)

    logger.info(f1_score(ytest, prediction, average="macro"))

    # Testing report
    logger.info("Scores report on test dataset:")
    logger.info(classification_report(ytest, prediction))

    # compute top_3 accuracy
    classes = list(text_classifier.classes_)
    # do mapping to classes indices
    ts = np.array([classes.index(i) for i in list(ytest)])
    logger.info("Top 3 accuracy on test dataset:")
    logger.info(utils.top_n_accuracy(prediction_proba, ts, 3))

    # Saving model
    logger.info("Saving model")
    dump(text_classifier, "saved_models/{}.joblib".format(config.exp_name))


if __name__ == "__main__":
    main()
