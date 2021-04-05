"""
Python API for FastText text classification model training.

We train a sentiment classifier using automatic hyperparameter optimization:
https://fasttext.cc/docs/en/autotune.html
"""
import argparse
import os
import fasttext


def trainer(filepath: str,
            train_file: str,
            valid_file: str,
            model_path: str,
            hyper_params: dict) -> None:
    """
    Train sentiment model using FastText automatic hyperparameter optimization:
    https://fasttext.cc/docs/en/autotune.html
    """
    train = os.path.join(filepath, train_file)
    valid = os.path.join(filepath, valid_file)
    model = fasttext.train_supervised(
        input=train,
        autotuneValidationFile=valid,
        **hyper_params,
    )
    os.makedirs(model_path, exist_ok=True)
    model.save_model(f"{model_path}.ftz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="Dataset path", default="../data/sst")
    parser.add_argument('--train', type=str, help="Training set filename", default="sst_train.txt")
    parser.add_argument('--valid', type=str, help="Validation set filename", default="sst_dev.txt")
    parser.add_argument('--model', type=str, help="Trained model path", default="../models/fasttext/model")
    parser.add_argument("--duration", type=int, default=1800, help="Autotuning duration in seconds")
    parser.add_argument("--modelsize", type=str, default="10M", help="Max allowed size for autotuned model file (`5M` means 5 MB)")
    parser.add_argument("--disable_verbose", action="store_true", help="Disable verbosity in autotuning output")

    args = parser.parse_args()

    # Run automatic hyperparameter tuning during training
    hyper_params = {
        "autotuneDuration": args.duration,
        "autotuneModelSize": args.modelsize,
        "verbose": 2 if args.disable_verbose else 3,  # '3' means verbose
    }

    trainer(args.filepath, args.train, args.valid, args.model, hyper_params)
