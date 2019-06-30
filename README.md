# Fine Grained Sentiment Classification
This repo shows a comparison and discussion of various NLP methods to perform 5-class sentiment classification on the  [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) (SST-5) dataset. To see the differences between the models and which classes they predict better or worse than one another, we train multiple types of rule-based, linear and neural network-based classifiers.

Currently the following classifiers have been implemented:
 - **TextBlob**: Rule-based, uses the internal `polarity` metric from the [TextBlob](https://textblob.readthedocs.io/en/dev/) library.
 - **Vader**: Rule-based, uses the `compound` polarity scores from the [VADER](https://www.nltk.org/_modules/nltk/sentiment/vader.html) library.
 - **Logistic Regression**: Trains a simple logistic regression model after converting the vocabulary to feature vectors and considering the effect of word frequencies using TF-IDF.
 - **SVM**: Trains a simple linear  support vector machine after converting the vocabulary to feature vectors and considering the effect of word frequencies using TF-IDF.
 - **FastText**: Trains a [FastText](https://fasttext.cc/docs/en/supervised-tutorial.html) classifier using a combination of tri-grams and a 7-word context window size.
 - **Flair**: Trains a [Flair NLP](https://github.com/zalandoresearch/flair) classifier using ["stacked" embeddings](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_7_TRAINING_A_MODEL.md), i.e. a combined representation of ELMo word embeddings with Flair (forward and backward) string embeddings.  

## Installation

First, set up virtual environment and install from ```requirements.txt```:

    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

For further development, simply activate the existing virtual environment.

    source venv/bin/activate


## Training the Classifiers

The training of the linear models (Logistic Regression and SVM) are done during runtime of the classifier (next step) since it requires very little tuning. To train the methods that rely on word/document embeddings, however, we use separate scripts to help more easily tune the hyperparameters. 

To train the FastText model, run `train_fasttext.py`. The following hyper-parameters gave the best results for the SST-5 dataset.

    python3 train_fasttext.py --lr 0.5 --epochs 100 --wordNgrams 3 --ws 7 --dim 100

To train the Flair model, run `train_flair.py`. This model takes significantly longer to run on a GPU-enabled machine. After several hours of training for about 25 epochs, decent results were observed with the below options (ELMo embeddings were used on top of Flair embeddings).

    python3 train_flair.py --lr 0.1 --epochs 25


## Run sentiment analysis and output confusion matrix
 
Once the classifiers have been trained on the SST-5 data, run the file ```run_classifiers.py``` to perform 5-class sentiment classification on the test set. This file accepts arguments for multiple classifier models at a time (just space-separate the model names, all in lower case). The requested classification models are run and evaluated on the test set and confusion matrix plots are output (for each model) in the `./Plots/` directory.

An example using simple rule-based or scikit-learn models are shown below.
 
    python3 run_classifiers.py --method textblob vader logistic svm

This will run the TextBlob, Vader, Logistic Regression and Linear SVM sentiment models sequentially, and output their classification results. 

To use a trained FastText/Flair for classification, the `--model` argument must be passed.

    python3 run_classifiers.py --method fasttext --model models/fasttext/sst.bin

OR 

    python3 run_classifiers.py --method flair --model models/flair/best-model.pt

    
To see a full list of optional arguments, type the following:  ```python3 run_classifiers.py --help```

    optional arguments:
    -h, --help            show this help message and exit
    --train TRAIN         Train data file (str)
    --dev DEV             Dev data file (str)
    --test TEST           Test/Validation data file (str)
    --method METHOD [METHOD ...]
                            Enter one or more methods (Choose from following:
                            textblob, vader, logistic, svm, fasttext, flair)
    --model MODEL         Trained classifier model file (str)
    --lower               Flag to convert test data strings to lower case (for
                            lower-case trained classifiers)

