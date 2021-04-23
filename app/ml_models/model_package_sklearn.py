from joblib import load
import sklearn
import pandas as pd
import numpy as np
import re


def preprocessing(text):
    # clean_ascii
    results = "".join(i for i in text if ord(i) < 128)
    return results.lower()


def keep_token(t):
    return t.is_alpha and not (t.is_space or t.is_punct or t.is_stop or t.like_num)


def lemmatize_doc(doc):
    """ Lemmatize a document """
    return [t.lemma_ for t in doc if keep_token(t)]


class ModelPackageSK:
    """
    Model package for Sklearn model
    """
    def __init__(self, weights_path, weights_name, spacy):
        """
        weights_path: path to the folder where sklearn model is saved
        weights_name: name of the file (model)
        spacy: a loaded spacy model 
        """

        # Load spacy NLP
        self.nlp = spacy

        # Load model
        try:
            self.model = load(f"{weights_path}/{weights_name}")

        except IOError:
            print("Error Loading the Model")
        # class mapping to indices
        self.classes = self.model.classes_

    def topk_predictions(self, text, k):
        """
        text: str, raw text input 
        k: int, to define top-k predictions
        returns a dict of k keys {classes: probability}
        """
        
        # Prepare input
        input_ = preprocessing(text)

        # Lemmatize
        clean_input = " ".join(lemmatize_doc(self.nlp(input_)))

        # Do inference, probabilities and top_k indices
        predictions = self.model.predict_proba([clean_input])
        best_k = np.argsort(predictions, axis=1)[:, -k:]

        # dictionnary of predicted classes with their probabilities
        results = {
            self.classes[i]: "{:12.2f}%".format(float(predictions[0][i]) * 100)
            for i in best_k[0][::-1]
        }

        return results
