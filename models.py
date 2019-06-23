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
    "Predict sentiment scores using TextBlob"
    def score(self, sentence: str) -> float:
        from textblob import TextBlob
        return TextBlob(sentence).sentiment.polarity

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
    "Predict sentiment scores using TextBlob"
    def __init__(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score(self, sentence: str) -> float:
        return self.vader.polarity_scores(sentence)['compound']

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
    "Predict sentiment scores using TextBlob"
    def __init__(self, fasttext_model):
        # pip install fasttext
        import fastText
        self.model = fastText.load_model(fasttext_model)

    def score(self, sentence: str) -> float:
        # Predict just the top label (hence 1 index below)
        labels, probabilities = self.model.predict(sentence, 1)
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


if __name__ == "__main__":
    fname = 'data/sst/sst_test.txt'
    # textblob(fname)
    # vader(fname)
    fasttext(fname, model='models/fasttext/sst.bin')
