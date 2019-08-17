#!/usr/bin/env python3
import argparse
from typing import Tuple
from pathlib import Path


def trainer(file_path: Path,
            filenames: Tuple[str, str, str],
            checkpoint: str,
            stack: str,
            n_epochs: int) -> None:
    """Train sentiment model using Flair NLP library:
    https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md

    To help provide added context, we can stack Glove, Bert or ELMo embeddings along with Flair embeddings.
    """
    # pip install flair allennlp
    from flair.datasets import ClassificationCorpus
    from flair.embeddings import FlairEmbeddings, DocumentRNNEmbeddings
    from flair.models import TextClassifier
    from flair.trainers import ModelTrainer
    from flair.training_utils import EvaluationMetric
    from flair.visual.training_curves import Plotter

    if stack == "glove":
        from flair.embeddings import WordEmbeddings
        stacked_embedding = WordEmbeddings('glove')
    elif stack == "elmo":
        from flair.embeddings import ELMoEmbeddings
        stacked_embedding = ELMoEmbeddings('original')
    elif stack == "bert":
        from flair.embeddings import BertEmbeddings
        stacked_embedding = BertEmbeddings('bert-base-cased')
    else:
        stacked_embedding = None

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

    # Stack Flair string-embeddings with optional embeddings
    word_embeddings = list(filter(None, [
        stacked_embedding,
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]))
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

    if not checkpoint:
        trainer = ModelTrainer(classifier, corpus)
    else:
        # If checkpoint file is defined, resume training
        checkpoint = classifier.load_checkpoint(Path(checkpoint))
        trainer = ModelTrainer.load_from_checkpoint(checkpoint, corpus)

    # Begin training (enable checkpointing to continue training at a later time, if desired)
    trainer.train(
        file_path,
        EvaluationMetric.MACRO_F1_SCORE,
        max_epochs=n_epochs,
        checkpoint=True
    )

    # Plot curves and store weights and losses
    plotter = Plotter()
    plotter.plot_training_curves(file_path / 'loss.tsv')
    plotter.plot_weights(file_path / 'weights.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, help="Dataset path", default="../data/sst")
    parser.add_argument('--train', type=str, help="Training set filename", default="sst_train.txt")
    parser.add_argument('--dev', type=str, help="Dev set filename", default="sst_dev.txt")
    parser.add_argument('--test', type=str, help="Test/validation set filename", default="sst_test.txt")
    parser.add_argument('--stack', type=str, help="Type of embeddings to stack along with Flair embeddings", default="glove")
    parser.add_argument('--epochs', type=int, help="Number of epochs", default=25)
    parser.add_argument('--checkpoint', type=str, help="Path of checkpoint file (to restart training)", default=None)

    args = parser.parse_args()

    if args.stack not in ['glove', 'elmo', 'bert']:
        parser.error("Please specify only one of glove, elmo or bert for stacked embeddings!")

    # Specify path and file names for train, dev and test data
    filepath = Path('./') / args.filepath
    filenames = (args.train, args.dev, args.test)
    trainer(filepath, filenames, args.checkpoint, stack=args.stack, n_epochs=args.epochs)
