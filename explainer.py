import lime.lime_text
import numpy as np
import webbrowser
from pathlib import Path


def tokenize_string(text):
    return text.split()


def fasttext_pred_formatted(model_file, text):
    """Generate an array of predicted labels/scores using FastText
    """
    import fasttext

    # Load fasttext classifier
    classifier = fasttext.load_model(model_file)
    labels, probs = classifier.predict(text, 5)

    # For each prediction, sort the probability scores in the same order for all texts
    # (fasttext by default returns predictions in the most likely order for each text)
    result = []
    for label, prob, txt in zip(labels, probs, text):
        order = np.argsort(np.array(label))
        result.append(prob[order])

    return np.array(result)


def flair_pred_formatted(model_file, text):
    """Generate an array of predicted labels/scores using the Flair NLP library
    """
    from flair.models import TextClassifier
    from flair.data import Sentence

    # Load flair model
    model = TextClassifier.load(model_file)
    # Make prediction and sort probabilities
    doc = Sentence(text)
    model.predict(doc, multi_class_prob=True)

    labels = [x.value for x in doc.labels]
    probs = np.array([x.score for x in doc.labels])

    # For each prediction, sort the probability scores in the same order for all texts
    # (flair by default returns predictions in the most likely order for each text)
    result = []
    for label, prob, txt in zip(labels, probs, text):
        order = np.argsort(np.array(label))
        result.append(prob[order])

    return np.array(result)


if __name__ == "__main__":
    # Example string
    sample = "The story loses its bite in a last-minute happy ending that 's even less plausible than the rest of the picture ."

    # Create a LimeTextExplainer
    explainer = lime.lime_text.LimeTextExplainer(
        # Specify split option
        split_expression=tokenize_string,
        # Our classifer uses bigrams (two-word pairs) to classify text, hence, order matters
        # and we cannot use bag of words.
        bow=False,
        # Specify class names for this case
        class_names=[1, 2, 3, 4, 5]
    )

    # Make a prediction and explain it:
    exp = explainer.explain_instance(
                    sample,
                    # classifier_fn=lambda x: fasttext_pred_formatted('models/fasttext/sst.bin', x),
                    classifier_fn=lambda x: flair_pred_formatted('models/flair/best-model-elmo.pt', x),
                    top_labels=1,
                    num_features=25,
                )

    output_filename = Path(__file__).parent / "explanation.html"
    exp.save_to_file(output_filename)

    # Open the explanation html in our web browser.
    # webbrowser.open(output_filename.as_uri())
