#!/usr/bin/env python3
import argparse
from typing import Tuple
from pathlib import Path

def trainer(file_path: Path,
            filenames: Tuple[str, str, str],
            n_epochs: int,
            learning_rate: float) -> None:
    """Train sentiment model using Flair NLP library:
    https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md

    To help provide added context, we stack ELMo embeddings along with Flair embeddings.
    """
    # pip install flair allennlp
    from flair.datasets import ClassificationCorpus
    from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings, ELMoEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer
    from flair.training_utils import EvaluationMetric
    from flair.visual.training_curves import Plotter
    # Define and Load corpus from the provided dataset
    train, dev, test = filenames
    corpus = ClassificationCorpus(
        file_path,
        train_file=train,
        dev_file=dev,
        test_file=test,
    )
    # Create label dictionary from provided labels in data
    label_dict = corpus.make_label_dictionary()
    # Stack Flair string-embeddings with ELMo word-embeddings (requires AllenNLP installed)
    word_embeddings = [
        ELMoEmbeddings('original'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]
    # Initialize document embedding by passing list of word embeddings
    document_embeddings = DocumentRNNEmbeddings(
        word_embeddings,
        hidden_size=512,
        reproject_words=True,
        reproject_words_dimension=256,
    )
    # Define classifier
    classifier = TextClassifier(
        document_embeddings,
        label_dictionary=label_dict,
        multi_label=False
    )
    trainer = ModelTrainer(classifier, corpus)
    # Begin training (enable checkpointing to continue training at a later time, if desired)
    trainer.train(
        file_path,
        EvaluationMetric.MACRO_F1_SCORE,
        learning_rate=0.1,
        max_epochs=n_epochs,
        checkpoint=True
    )
    # Plot curves and store weights and losses
    plotter = Plotter()
    plotter.plot_training_curves(file_path / 'loss.tsv')
    plotter.plot_weights(file_path / 'weights.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="Dataset path", default="data/sst")
    parser.add_argument('--train', type=str, help="Training set filename", default="sst_train.txt")
    parser.add_argument('--dev', type=str, help="Dev set filename", default="sst_dev.txt")
    parser.add_argument('--test', type=str, help="Test/validation set filename", default="sst_test.txt")
    parser.add_argument('--epochs', type=int, help="Number of epochs", default=10)
    parser.add_argument('--lr', type=float, help="Starting learning rate", default=0.1)

    args = parser.parse_args()

    # Specify path and file names for train, dev and test data
    filepath = Path('./') / args.filepath
    filenames = (args.train, args.dev, args.test)
    trainer(filepath, filenames, n_epochs=args.epochs, learning_rate=args.lr)
