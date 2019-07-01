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

The method class and classifier model specification can be done using the dictionary within the file `run_classifiers.py`. The rule-based and linear models do not have trained models associated with them, hence they are left as `None`. For any other learning-based models, the trained model file can be specified in this dictionary to avoid having to pass it as an argument. 

    METHODS = {
        'textblob': {
            'class': TextBlobSentiment,
            'model': None
        },
        'vader': {
            'class': VaderSentiment,
            'model': None
        },
        'logistic': {
            'class': LogisticRegressionSentiment,
            'model': None
        },
        'svm': {
            'class': SVMSentiment,
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

The above dictionary makes it easier to update the framework with more models and methods over time - simply update the method name and its class names and models files as shown above. 

To run a single case just pass one method as an argument
 
    python3 run_classifiers.py --method textblob

All methods from the dictionary can be run sequentially as follows:

    python3 run_classifiers.py --method textblob vader logistic svm fasttext flair

If at a later time, multiple versions of trained models need to be run sequentially, we can specify these using the `--model` argument.

    python3 run_classifiers.py --method fasttext --model models/fasttext/sst-bigram.bin
    python3 run_classifiers.py --method fasttext --model models/fasttext/sst-trigram.bin

OR 

    python3 run_classifiers.py --method flair --model models/flair/best-model-elmo.pt
    python3 run_classifiers.py --method flair --model models/flair/best-model-glove.pt



