#!/usr/bin/env python3

from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, ELMoEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter
from pathlib import Path

# Specify path to train and test data
file_path = Path('./') / 'data/sst'

train = 'sst_train.txt'
test = 'sst_test.txt'
dev = 'sst_dev.txt'

# Load corpus
corpus = ClassificationCorpus(file_path,
                              train_file=train,
                              dev_file=dev,
                              test_file=test,
                              )
# Create label dictionary
label_dict = corpus.make_label_dictionary()
# Specify word embeddings to use (can combine multiple types)
word_embeddings = [ELMoEmbeddings('original'),
                   FlairEmbeddings('news-forward'),
                   FlairEmbeddings('news-backward'),
                   ]
# Initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, 
                                            reproject_words=True, reproject_words_dimension=256)

# Define classifier
classifier = TextClassifier(document_embeddings,
                            label_dictionary=label_dict,
                            multi_label=False)
trainer = ModelTrainer(classifier, corpus)

# Begin training
trainer.train(file_path,
              EvaluationMetric.MACRO_F1_SCORE,
              max_epochs=10,
              checkpoint=True
             )

plotter = Plotter()
plotter.plot_training_curves(file_path / 'loss.tsv')
plotter.plot_weights(file_path / 'weights.txt')

