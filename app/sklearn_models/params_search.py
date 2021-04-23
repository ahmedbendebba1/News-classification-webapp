"""
Search for SVM
Run : 
balanced data: python3 params_search.py --model svc --exp_name svmsearch --data dataset_processed_balanced
all data: python3 params_search.py --model svc --exp_name svmsearch_all --data dataset_processed
Search for Logisitic Regression
Run : 
balanced: python3 params_search.py --model lreg --exp_name logreg --data dataset_processed_balanced
all: python3 params_search.py --model lreg --exp_name logreg_all --data dataset_processed
"""
import pandas as pd
import utils
import argparse
import os
from functools import partial
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Absolute PATH where code and data folder are
PATH = ".."

# Params space for grid_search
PARAMS_SPACE = {
    "tfidf__ngram_range": [(1, 2), (1, 3)],
    "clf__C": (0.1, 0.2, 0.3, 0.5, 1, 5),
}

MODELS = {"svc": LinearSVC, "lreg": LogisticRegression}


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(
        name, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=" ")
    return parser


class ModelSearch(argparse.Namespace):  # noqa
    """
    Model search configuration
    """
    def build_parser(self):
        parser = get_parser("Params Search")
        parser.add_argument("--model", required=True)
        parser.add_argument("--exp_name", required=True)
        parser.add_argument("--score", default="f1")
        parser.add_argument("--data", default="dataset_processed_balanced")

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))


# Define config
config = ModelSearch()

# Define logger
logger = utils.get_logger("logs/search_{}.log".format(config.exp_name))


def main():
    # Loading Dataset
    logger.info("Start Loading saved feather dataset")
    data = pd.read_feather(f"{PATH}/data/{config.data}")
    logger.info("Data loaded.")

    # TF-IDF transformer + simple algorithm
    logger.info("Define the pipeline:")
    text_classifier = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_df=0.9, min_df=3, analyzer="word", sublinear_tf=True
                ),
            ),
            ("clf", MODELS[config.model]()),
        ]
    )
    # Define Cross validation grid search
    grid_search = GridSearchCV(
        text_classifier,
        PARAMS_SPACE,
        scoring="%s_macro" % config.score,
        cv=5,
        n_jobs=-1,
    )

    # Split data : training/testing
    logger.info("Training, testing split.")
    xtrain, xtest, ytrain, ytest = train_test_split(
        data["clean_text"], data.category, random_state=42, test_size=0.1
    )

    # Train model and evaluate
    logger.info("Searching for best params...")
    grid_search.fit(xtrain, ytrain)

    # Scoring
    logger.info("Best parameters set found on development set:")
    logger.info(grid_search.best_params_)

    logger.info("Grid scores on development set:")
    means = grid_search.cv_results_["mean_test_score"]
    stds = grid_search.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
        logger.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # Testing report
    logger.info("Scores report on test dataset using best model:")
    prediction = grid_search.predict(xtest)
    logger.info(classification_report(ytest, prediction))

    # Saving best model
    # logger.info("Saving best model")
    # dump(grid_search.best_estimator_, 'saved_models/search-{}.joblib'.format(config.exp_name))


if __name__ == "__main__":
    main()
