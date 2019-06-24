from typing import Tuple
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class Utils:
    """Class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def test_data(self, fname: str,
                  colnames=['truth', 'sentence']) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['truth'] = df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')
        return df

    def accuracy(self, df: pd.DataFrame, pred_column: str) -> Tuple[float, float]:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['truth'], df[pred_column])*100
        f1 = f1_score(df['truth'], df[pred_column], average='macro')
        return acc, f1


class TextBlobSentiment(Utils):
    """Predict sentiment scores using TextBlob.
    https://textblob.readthedocs.io/en/dev/
    """
    def score(self, text: str) -> float:
        # pip install textblob
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

    def predict(self, fname: str, pred_column: str) -> pd.DataFrame:
        df = super().test_data(fname)
        df['score'] = df['sentence'].apply(self.score)
        # Convert float score to category based on binning
        df[pred_column] = pd.cut(df['score'],
                                 bins=5,
                                 labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df

    def accuracy(self, df: pd.DataFrame, pred_column: str) -> Tuple[float, float]:
        "Return accuracy metrics"
        acc, f1 = super().accuracy(df, pred_column)
        print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))
        return acc, f1


class VaderSentiment(Utils):
    """Predict sentiment scores using Vader.
    Tested using nltk.sentiment.vader and Python 3.6+
    https://www.nltk.org/_modules/nltk/sentiment/vader.html
    """
    def __init__(self) -> None:
        # pip install nltk
        # python > import nltk > nltk.download() > d > vader_lexicon
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return self.vader.polarity_scores(text)['compound']

    def predict(self, fname: str, pred_column: str) -> pd.DataFrame:
        df = super().test_data(fname)
        df['score'] = df['sentence'].apply(self.score)
        # Convert float score to category based on binning
        df[pred_column] = pd.cut(df['score'],
                                 bins=5,
                                 labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df

    def accuracy(self, df: pd.DataFrame, pred_column: str) -> Tuple[float, float]:
        "Return accuracy metrics"
        acc, f1 = super().accuracy(df, pred_column)
        print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))
        return acc, f1


class FastTextSentiment(Utils):
    """Predict sentiment scores using FastText.
    https://fasttext.cc/
    """
    def __init__(self, fasttext_model: str) -> None:
        # pip install fasttext
        import fastText
        try:
            self.model = fastText.load_model(fasttext_model)
        except ValueError:
            raise Exception("Cannot find specified model file: '{}'.".format(fasttext_model))

    def score(self, text: str) -> float:
        # Predict just the top label (hence 1 index below)
        labels, probabilities = self.model.predict(text, 1)
        pred = int(labels[0][-1])
        return pred

    def predict(self, fname: str, pred_column: str) -> pd.DataFrame:
        df = super().test_data(fname)
        df[pred_column] = df['sentence'].str.lower().apply(self.score)
        return df

    def accuracy(self, df: pd.DataFrame, pred_column: str) -> Tuple[float, float]:
        "Return accuracy metrics"
        acc, f1 = super().accuracy(df, pred_column)
        print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))
        return acc, f1


class FlairSentiment(Utils):
    """Predict sentiment scores using Flair.
    https://github.com/zalandoresearch/flair
    Tested on Flair version 0.4.2+ and Python 3.6+
    """
    def __init__(self, flair_model: str) -> None:
        "Use the latest version of Flair NLP from their GitHub repo!"
        # pip install flair
        # pip install tqdm
        from flair.models import TextClassifier
        try:
            self.model = TextClassifier.load(flair_model)
        except ValueError:
            raise Exception("Cannot find specified model file: '{}'.".format(flair_model))

    def score(self, text: str) -> float:
        from flair.data import Sentence
        doc = Sentence(text)
        self.model.predict(doc)
        pred = int(doc.labels[0].value)
        return pred

    def predict(self, fname: str, pred_column: str) -> pd.DataFrame:
        "Use tqdm to display model prediction status bar"
        from tqdm import tqdm
        tqdm.pandas()
        df = super().test_data(fname)
        df[pred_column] = df['sentence'].progress_apply(self.score)
        return df

    def accuracy(self, df: pd.DataFrame, pred_column: str) -> Tuple[float, float]:
        "Return accuracy metrics"
        acc, f1 = super().accuracy(df, pred_column)
        print("Accuracy: {}\nMacro F1-score: {}".format(acc, f1))
        return acc, f1


# Run methods
def textblob(fname: str, pred_column='textblob_pred') -> None:
    "Run TextBlob sentiment model"
    tb = TextBlobSentiment()
    result = tb.predict(fname, pred_column)
    _ = tb.accuracy(result, pred_column)


def vader(fname: str, pred_column='vader_pred') -> None:
    "Run Vader sentiment model"
    va = VaderSentiment()
    result = va.predict(fname, pred_column)
    _ = va.accuracy(result, pred_column)


def fasttext(fname: str, model: str, pred_column='fasttext_pred') -> None:
    "Run FastText sentiment model"
    ft = FastTextSentiment(model)
    result = ft.predict(fname, pred_column)
    _ = ft.accuracy(result, pred_column)


def flair(fname: str, model: str, pred_column='flair_pred') -> None:
    "Run Flair sentiment model"
    fl = FlairSentiment(model)
    result = fl.predict(fname, pred_column)
    _ = fl.accuracy(result, pred_column)


if __name__ == "__main__":
    fname = 'data/sst/sst_test.txt'
    textblob(fname)
    # vader(fname)
    # fasttext(fname, model='models/fasttext/sst.bin')
    # flair(fname, "models/flair/best-model.pt")
