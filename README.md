# Fine Grained Sentiment Analysis: An Overview of Different Methods
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


## Run sentiment analysis and output confusion matrix plots
 
Run the file ```run_classifiers.py``` to perform 5-class sentiment classification. This file accepts arguments for multiple classifier models at a time (just space-separate the model names, all in lower case). The requested classification models are run on the test set, and confusion matrix plots are output (for each model) in the `./Plots/` directory.

An example using simple rule-based or scikit-learn models are shown below.
 
    python3 run_classifiers.py --method textblob vader logistic vader

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

