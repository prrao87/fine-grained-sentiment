"""
Python API for FastText text classification model training.

All default arguments in this file are obtained using hyperparameter autotuning
in fastText: https://fasttext.cc/docs/en/autotune.html

Run the following command for the CLI utility to obtain optimum hyperparameters:

./fasttext supervised -input sst_train.txt -output model \
-autotune-validation sst_dev.txt -autotune-modelsize 10M -verbose 3
"""

import argparse
import os
import fasttext


def make_dirs(dirpath: str) -> None:
    """Make directories for output if necessary"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def trainer(filepath: str,
            train_file: str,
            model_path: str,
            hyper_params: dict) -> None:
    """Train sentiment model using FastText:
    https://fasttext.cc/docs/en/supervised-tutorial.html
    """
    train = os.path.join(filepath, train_file)
    model = fasttext.train_supervised(input=train, **hyper_params)
    print("FastText model trained with hyperparameters: \n {}".format(hyper_params))
    # Save models to model directory for fasttext
    make_dirs(model_path)
    model.save_model(os.path.join(model_path, "sst5_hyperopt.bin"))
    # Quantize model to reduce space usage
    model.quantize(input=train, qnorm=True, retrain=True, cutoff=110539)
    model.save_model(os.path.join(model_path, "sst5_hyperopt.ftz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="Dataset path", default="../data/sst")
    parser.add_argument('--train', type=str, help="Training set filename", default="sst_train.txt")
    parser.add_argument('--model', type=str, help="Trained model path", default="../models/fasttext")
    parser.add_argument('--epochs', type=int, help="Number of epochs", default=100)
    parser.add_argument('--lr', type=float, help="Learning rate", default=0.35)
    parser.add_argument('--ws', type=int, help="Size of the context window", default=5)
    parser.add_argument('--wordNgrams', type=int, help="Max length of word ngram", default=3)
    parser.add_argument('--dim', type=int, help="Size of the word vectors", default=155)
    parser.add_argument('--minn', type=int, help="Min length of char ngram", default=2)
    parser.add_argument('--maxn', type=int, help="Max length of char ngram", default=5)
    parser.add_argument('--bucket', type=int, help="Number of buckets", default=2014846)

    args = parser.parse_args()

    # Use fastText command line utility to run hyperparameter optimization for best default args!
    hyper_params = {
        "lr": args.lr,
        "epoch": args.epochs,
        "wordNgrams": args.wordNgrams,
        "dim": args.dim,
        "ws": args.ws,
        "minn": args.minn,
        "maxn": args.maxn,
        "bucket": args.bucket,
    }

    trainer(args.filepath, args.train, args.model, hyper_params)
