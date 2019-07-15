import argparse
import lime.lime_text
import numpy as np
from pathlib import Path


METHODS = {
    'fasttext': "models/fasttext/sst.bin",
    'flair': "models/flair/best-model-elmo.pt"
}


def load_fasttext(path_to_model):
    "Load FastText classifier"
    import fasttext
    classifier = fasttext.load_model(path_to_model)
    return classifier


def load_flair(path_to_model):
    "Load Flair classifier"
    from flair.models import TextClassifier
    classifier = TextClassifier.load(path_to_model)
    return classifier


def fasttext_predictor(classifier, texts):
    """Generate an array of predicted labels/scores using FastText
    """
    labels, probs = classifier.predict(texts, 5)

    # For each prediction, sort the probability scores in the same order for all texts
    result = []
    for label, prob, text in zip(labels, probs, texts):
        order = np.argsort(np.array(label))
        result.append(prob[order])

    return np.array(result)


def flair_predictor(classifier, texts):
    """Generate an array of predicted labels/scores using the Flair NLP library
    """
    from flair.data import Sentence
    # Make prediction and sort probabilities
    labels = []
    probs = []
    for text in texts:
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


def main(method, path_to_model, text):
    """Run LIME explainer on provided classifier
    """
    if method == "fasttext":
        classifier = load_fasttext(path_to_model)
        predictor = fasttext_predictor
    elif method == "flair":
        classifier = load_flair(path_to_model)
        predictor = flair_predictor
    else:
        raise Exception("Requested method {} explainer funtction not implemented!")

    # Create a LimeTextExplainer
    explainer = lime.lime_text.LimeTextExplainer(
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

    # Example string
    sample = "It 's not horrible , just horribly mediocre ."

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
            parser.error("Please choose from existing methods! {}".format(", ".join(method_list)))
        try:
            path_to_model = METHODS[method]
            # Run explainer function
            exp = main(method, path_to_model, sample)

            # Output to HTML
            output_filename = Path(__file__).parent / "explanation-{}.html".format(method)
            exp.save_to_file(output_filename)
        except Exception as e:
            print(e)
