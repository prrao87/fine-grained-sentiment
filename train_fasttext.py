import argparse
import os
import fasttext


def trainer(filepath: str,
            train_file: str,
            model_file: str,
            hyper_params: dict) -> None:
    """Train sentiment model using FastText:
    https://fasttext.cc/docs/en/supervised-tutorial.html
    """
    train = os.path.join(filepath, train_file)
    output = os.path.join(filepath, model_file)
    model = fasttext.train_supervised(input=train, **hyper_params)
    print("FastText model trained with hyperparameters: \n {}".format(hyper_params))
    # Save model
    model.save_model(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="Dataset path", default="data/sst")
    parser.add_argument('--train', type=str, help="Training set filename", default="sst_train.txt")
    parser.add_argument('--model', type=str, help="Trained model output name", default="sst.bin")
    parser.add_argument('--epochs', type=int, help="Number of epochs", default=100)
    parser.add_argument('--lr', type=float, help="Learning rate", default=0.5)
    parser.add_argument('--ws', type=int, help="Size of the context window", default=3)
    parser.add_argument('--wordNgrams', type=int, help="Max length of word ngram", default=3)
    parser.add_argument('--dim', type=int, help="Size of the word vectors", default=100)

    args = parser.parse_args()

    hyper_params = {
        "lr": args.lr,
        "epoch": args.epochs,
        "wordNgrams": args.wordNgrams,
        "dim": args.dim,
        "ws": args.ws,
    }

    trainer(args.filepath, args.train, args.model, hyper_params)
