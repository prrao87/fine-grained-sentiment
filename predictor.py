"""
Run fine-grained sentiment classifier based on chosen method
"""
import argparse
import os
from typing import Tuple, Any
from plotter import plot_confusion_matrix
from classifiers import *

# Path to train & test data
TRAIN_PATH = "data/sst/sst_train.txt"
DEV_PATH = "data/sst/sst_dev.txt"
TEST_PATH = "data/sst/sst_test.txt"

# List of currently implemented sentiment classification methods
METHODS = {
    'textblob': {
        'class': "TextBlobSentiment",
        'model': None
    },
    'vader': {
        'class': "VaderSentiment",
        'model': None
    },
    'logistic': {
        'class': "LogisticRegressionSentiment",
        'model': None
    },
    'svm': {
        'class': "SVMSentiment",
        'model': None
    },
    'fasttext': {
        'class': "FastTextSentiment",
        'model': "models/fasttext/sst5_hyperopt.ftz"
    },
    'flair': {
        'class': "FlairSentiment",
        'model': "models/flair/best-model-elmo.pt"
    },
    'transformer': {
        'class': "TransformerSentiment",
        'model': "models/transformer",
    }
}


def get_class(method: str, filename: str) -> Any:
    "Instantiate class using its string name"
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_(filename)


def make_dirs(dirpath: str) -> None:
    """Make directories for output if necessary"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def run_classifier(files: Tuple[str, str, str],
                   method: str,
                   method_class: Base,
                   model_file: str,
                   lower_case: bool) -> None:
    "Inherit classes from classifiers.py and apply the predict/accuracy methods"
    train, dev, test = files  # Unpack train, dev and test filenames
    result = method_class.predict(train, test, lower_case)
    method_class.accuracy(result)
    # Plot confusion matrix
    make_dirs("Plots")
    fig, ax = plot_confusion_matrix(result['truth'], result['pred'], normalize=True)
    ax.set_title("Normalized Confusion Matrix: {}".format(method.title()))
    fig.tight_layout()
    fig.savefig("Plots/{}.png".format(method))


if __name__ == "__main__":
    # Get list of available methods:
    method_list = [method for method in METHODS.keys()]
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Train data file (str)", default=TRAIN_PATH)
    parser.add_argument('--dev', type=str, help="Dev data file (str)", default=DEV_PATH)
    parser.add_argument('--test', type=str, help="Test/Validation data file (str)", default=TEST_PATH)
    parser.add_argument('--method', type=str, nargs='+', help="Enter one or more methods \
                        (Choose from following: {})".format(", ".join(method_list)),
                        required=True)
    parser.add_argument('--model', type=str, help="Trained classifier model file or path (str)", default=None)
    parser.add_argument('--lower', action="store_true", help="Flag to convert test data strings \
                        to lower case (for lower-case trained classifiers)")
    args = parser.parse_args()

    files = (args.train, args.dev, args.test)   # Paths to train, dev and test files (str)
    lower_case = args.lower
    for method in args.method:
        if method not in METHODS.keys():
            parser.error("Please choose from existing methods! {}".format(", ".join(method_list)))
        try:
            if args.model:
                model_file = args.model
            else:
                model_file = METHODS[method]['model']
            # Instantiate the implemented classifier class
            method_class = get_class(method, model_file)
        except KeyError:
            raise Exception("Incorrect method specification. Please choose from existing methods!\n{}"
                            .format(", ".join(method_list)))

        print("--\nRunning {} classifier on test data.".format(method))
        run_classifier(files, method, method_class, model_file, lower_case)
