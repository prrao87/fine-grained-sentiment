"""
Run fine-grained sentiment classifier based on chosen method
"""
import argparse
import pandas as pd
from models import *

TEST_PATH = "data/sst/sst_test.txt"

# List of currently implemented sentiment classification methods
METHODS = {
    'textblob': {
        'class': TextBlobSentiment,
        'model': None
    },
    'vader': {
        'class': VaderSentiment,
        'model': None
    },
    'fasttext': {
        'class': FastTextSentiment,
        'model': "models/fasttext/sst.bin"
    },
    'flair': {
        'class': FlairSentiment,
        'model': "models/flair/best-model.pt"
    },
}


def run_classifier(fname: str,
                   method_class: Base,
                   model_file: str,
                   lower_case: bool) -> pd.DataFrame:
    "Inherit classes from models.py and apply the predict/accuracy methods"
    result = method_class.predict(fname, lower_case)
    method_class.accuracy(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, help="Path (str) to test data file", default=TEST_PATH)
    parser.add_argument('--method', type=str, help="Sentiment classifier method", default="textblob")
    parser.add_argument('--model', type=str, help="Path (str) to trained classifier model file", default=None)
    parser.add_argument('--lower', action="store_true", help="Flag to convert test data strings \
                        to lower case (for lower-case trained classifiers)")
    args = parser.parse_args()

    fname = args.test   # Test file path (str)
    model_file = METHODS[args.method]['model']
    method_class = METHODS[args.method]['class'](model_file)   # Instantiate the implemented classifier class
    lower_case = args.lower

    res = run_classifier(fname, method_class, model_file, lower_case)