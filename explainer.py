import argparse
import numpy as np
import sklearn.pipeline
from pathlib import Path
from typing import List, Any
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

METHODS = {
    'logistic': {
        'class': "LogisticExplainer",
        'file': "data/sst/sst_train.txt"
    },
    'fasttext': {
        'class': "FastTextExplainer",
        'file': "models/fasttext/sst.bin"
    },
    'flair': {
        'class': "FlairExplainer",
        'file': "models/flair/best-model-elmo.pt"
    },
}


def explainer_class(method: str, filename: str) -> Any:
    "Instantiate class using its string name"
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_(filename)


class LogisticExplainer:
    """Class to explain classification results of a scikit-learn 
    Logistic Regression Pipeline. The model is trained within this class."""
    def __init__(self, path_to_train_data: str) -> None:
        import pandas as pd
        # Read in training data set
        self.train_df = pd.read_csv(path_to_train_data, sep='\t', header=None, names=["truth", "text"])
        self.train_df['truth'] = self.train_df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        self.train_df['truth'] = self.train_df['truth'].astype(int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        "Create sklearn logistic regression model pipeline"
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
            ]
        )
        # Train model
        classifier = pipeline.fit(self.train_df['text'], self.train_df['truth'])
        return classifier

    def predict(self, texts: List[str]) -> np.array:
        """Generate an array of predicted scores (probabilities) from sklearn
        Logistic Regression Pipeline."""
        classifier = self.train()
        probs = classifier.predict_proba(texts)
        return probs


class FastTextExplainer:
    """Class to explain classification results of FastText.
    Assumes that we already have a trained FastText model with which to make predictions."""
    def __init__(self, path_to_model: str) -> None:
        import fasttext
        self.classifier = fasttext.load_model(path_to_model)

    def predict(self, texts: List[str]) -> np.array:
        "Generate an array of predicted scores using the FastText"
        labels, probs = self.classifier.predict(texts, 5)

        # For each prediction, sort the probability scores in the same order for all texts
        result = []
        for label, prob, text in zip(labels, probs, texts):
            order = np.argsort(np.array(label))
            result.append(prob[order])
        return np.array(result)


class FlairExplainer:
    """Class to explain classification results of Flair.
    Assumes that we already have a trained Flair model with which to make predictions."""
    def __init__(self, path_to_model: str) -> None:
        from flair.models import TextClassifier
        self.classifier = TextClassifier.load(path_to_model)

    def predict(self, texts: List[str]) -> np.array:
        "Generate an array of predicted scores using the Flair NLP library"
        from flair.data import Sentence
        labels, probs = [], []
        for text in tqdm(texts):
            # Iterate through text list and make predictions
            doc = Sentence(text)
            self.classifier.predict(doc, multi_class_prob=True)
            labels.append([x.value for x in doc.labels])
            probs.append([x.score for x in doc.labels])
        probs = np.array(probs)   # Convert probabilities to Numpy array

        # For each prediction, sort the probability scores in the same order for all texts
        result = []
        for label, prob, text in zip(labels, probs, texts):
            order = np.argsort(np.array(label))
            result.append(prob[order])
        return np.array(result)


def main(method: str,
         path_to_file: str,
         text: str) -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""

    model = explainer_class(method, path_to_file)
    predictor = model.predict

    # Create a LimeTextExplainer
    explainer = LimeTextExplainer(
        # Specify split option
        split_expression=lambda x: x.split(),
        # Our classifer uses bigrams or contextual ordering to classify text
        # Hence, order matters, and we cannot use bag of words.
        bow=False,
        # Specify class names for this case
        class_names=[1, 2, 3, 4, 5]
    )

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
        text,
        classifier_fn=predictor,
        top_labels=1,
        num_features=10,
    )
    return exp


if __name__ == "__main__":
    # Evaluation text
    samples = [
        "It 's not horrible , just horribly mediocre .",
        "Light , cute and forgettable .",
    ]

    # Get list of available methods:
    method_list = [method for method in METHODS.keys()]
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, nargs='+', help="Enter one or more methods \
                        (Choose from following: {})".format(", ".join(method_list)),
                        required=True)

    args = parser.parse_args()

    for method in args.method:
        if method not in METHODS.keys():
            parser.error("Please choose from the below existing methods! \n{}".format(", ".join(method_list)))
        path_to_file = METHODS[method]['file']
        # Run explainer function
        for i, text in enumerate(samples):
            exp = main(method, path_to_file, text)

            # Output to HTML
            output_filename = Path(__file__).parent / "{}-explanation-{}.html".format(i+1, method)
            exp.save_to_file(output_filename)
            print("Output explainer data {} to HTML".format(i+1))
