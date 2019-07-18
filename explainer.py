import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Any
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

## @TODO: Clean up code structure and make it more modular

METHODS = {
    'logistic': "data/sst/sst_train.txt",
    'fasttext': "models/fasttext/sst.bin",
    'flair': "models/flair/best-model-elmo.pt"
}


def load_logistic(path_to_train_data: str) -> Any:
    "Train a logistic regression classifier"
    # Read in and transform trainind data
    train_df = pd.read_csv(path_to_train_data, sep='\t', header=None, names=["truth", "text"])
    train_df['truth'] = train_df['truth'].str.replace('__label__', '')
    # Categorical data type for truth labels
    train_df['truth'] = train_df['truth'].astype(int).astype('category')

    # Create sklearn logistic regression model pipeline
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
    learner = pipeline.fit(train_df['text'], train_df['truth'])
    return learner


def load_fasttext(path_to_model: str) -> Any:
    "Load FastText classifier"
    import fasttext
    classifier = fasttext.load_model(path_to_model)
    return classifier


def load_flair(path_to_model: str) -> Any:
    "Load Flair classifier"
    from flair.models import TextClassifier
    classifier = TextClassifier.load(path_to_model)
    return classifier


def logistic_predictor(classifier: Any, texts: List[str]) -> np.array:
    """Generate an array of predicted scores (probabilities) from sklearn
    Logistic Regression Pipeline."""
    probs = classifier.predict_proba(texts)
    return probs


def fasttext_predictor(classifier: Any, texts: List[str]) -> np.array:
    """Generate an array of predicted labels/scores using FastText
    """
    labels, probs = classifier.predict(texts, 5)

    # For each prediction, sort the probability scores in the same order for all texts
    result = []
    for label, prob, text in zip(labels, probs, texts):
        order = np.argsort(np.array(label))
        result.append(prob[order])

    return np.array(result)


def flair_predictor(classifier: Any, texts: List[str]) -> np.array:
    """Generate an array of predicted labels/scores using the Flair NLP library
    """
    from flair.data import Sentence
    labels = []
    probs = []
    for text in tqdm(texts):
        # Iterate through text list and make predictions
        doc = Sentence(text)
        classifier.predict(doc, multi_class_prob=True)
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
    """Run LIME explainer on provided classifier
    """
    if method == "logistic":
        classifier = load_logistic(path_to_file)
        predictor = logistic_predictor
    elif method == "fasttext":
        classifier = load_fasttext(path_to_file)
        predictor = fasttext_predictor
    elif method == "flair":
        classifier = load_flair(path_to_file)
        predictor = flair_predictor
    else:
        raise Exception("Requested method {} explainer function not implemented!")

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
        classifier_fn=lambda x: predictor(classifier, x),
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
        try:
            path_to_file = METHODS[method]
            # Run explainer function
            for i, text in enumerate(samples):
                exp = main(method, path_to_file, text)

                # Output to HTML
                output_filename = Path(__file__).parent / "{}-explanation-{}.html".format(i, method)
                exp.save_to_file(output_filename)
                print("Output explainer data {} to HTML".format(i+1))
        except Exception as e:
            print(e)