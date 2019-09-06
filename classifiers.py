import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class Base:
    """Base class that houses common utilities for reading in test data
    and calculating model accuracy and F1 scores.
    """
    def __init__(self) -> None:
        pass

    def read_data(self, fname: str, lower_case: bool=False,
                  colnames=['truth', 'text']) -> pd.DataFrame:
        "Read in test data into a Pandas DataFrame"
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['truth'] = df['truth'].str.replace('__label__', '')
        # Categorical data type for truth labels
        df['truth'] = df['truth'].astype(int).astype('category')
        # Optional lowercase for test data (if model was trained on lowercased text)
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def accuracy(self, df: pd.DataFrame) -> None:
        "Prediction accuracy (percentage) and F1 score"
        acc = accuracy_score(df['truth'], df['pred'])*100
        f1 = f1_score(df['truth'], df['pred'], average='macro')*100
        print("Accuracy: {:.3f}\nMacro F1-score: {:.3f}".format(acc, f1))


class TextBlobSentiment(Base):
    """Predict sentiment scores using TextBlob.
    https://textblob.readthedocs.io/en/dev/
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()

    def score(self, text: str) -> float:
        # pip install textblob
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        df = self.read_data(test_file, lower_case)
        df['score'] = df['text'].apply(self.score)
        # Convert float score to category based on binning
        df['pred'] = pd.cut(df['score'],
                            bins=5,
                            labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df


class VaderSentiment(Base):
    """Predict sentiment scores using Vader.
    Tested using nltk.sentiment.vader and Python 3.6+
    https://www.nltk.org/_modules/nltk/sentiment/vader.html
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install nltk
        # python > import nltk > nltk.download() > d > vader_lexicon
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return self.vader.polarity_scores(text)['compound']

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        df = self.read_data(test_file, lower_case)
        df['score'] = df['text'].apply(self.score)
        # Convert float score to category based on binning
        df['pred'] = pd.cut(df['score'],
                            bins=5,
                            labels=[1, 2, 3, 4, 5])
        df = df.drop('score', axis=1)
        return df


class LogisticRegressionSentiment(Base):
    """Predict sentiment scores using Logistic Regression.
    Uses a sklearn pipeline.
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install sklearn
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(
                    solver='newton-cg',
                    multi_class='multinomial',
                    random_state=42,
                    max_iter=100,
                )),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        # Fit the learner to the test data
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class SVMSentiment(Base):
    """Predict sentiment scores using a linear Support Vector Machine (SVM).
    Uses a sklearn pipeline.
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install sklearn
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='hinge',
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    learning_rate='optimal',
                    tol=None,
                )),
            ]
        )

    def predict(self, train_file: str, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Train model using sklearn pipeline"
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        # Fit the learner to the test data
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class FastTextSentiment(Base):
    """Predict sentiment scores using FastText.
    https://fasttext.cc/
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        # pip install fasttext
        import fasttext
        try:
            self.model = fasttext.load_model(model_file)
        except ValueError:
            raise Exception("Please specify a valid trained FastText model file (.bin or .ftz extension)'{}'."
                            .format(model_file))

    def score(self, text: str) -> int:
        # Predict just the top label (hence 1 index below)
        labels, probabilities = self.model.predict(text, 1)
        pred = int(labels[0][-1])
        return pred

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].apply(self.score)
        return df


class FlairSentiment(Base):
    """Predict sentiment scores using Flair.
    https://github.com/zalandoresearch/flair
    Tested on Flair version 0.4.2+ and Python 3.6+
    """
    def __init__(self, model_file: str=None) -> None:
        super().__init__()
        "Use the latest version of Flair NLP from their GitHub repo!"
        # pip install flair
        from flair.models import TextClassifier
        try:
            self.model = TextClassifier.load(model_file)
        except ValueError:
            raise Exception("Please specify a valid trained Flair PyTorch model file (.pt extension)'{}'."
                            .format(model_file))

    def score(self, text: str) -> int:
        from flair.data import Sentence
        doc = Sentence(text)
        self.model.predict(doc)
        pred = int(doc.labels[0].value)
        return pred

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Use tqdm to display model prediction status bar"
        # pip install tqdm
        from tqdm import tqdm
        tqdm.pandas()
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].progress_apply(self.score)
        return df


class TransformerSentiment(Base):
    """Predict sentiment scores using a causal transformer.
    Code for training/evaluating the transformer is as per the NAACL transfer learning repository.
    https://github.com/huggingface/naacl_transfer_learning_tutorial
    """
    def __init__(self, model_path: str=None) -> None:
        super().__init__()
        "Requires the BertTokenizer from pytorch_transformers"
        # pip install pytorch_transformers
        import os
        import torch
        from pytorch_transformers import BertTokenizer, cached_path
        from training.transformer_utils.model import TransformerWithClfHeadAndAdapters
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.config = torch.load(cached_path(os.path.join(model_path, "model_training_args.bin")))
            self.model = TransformerWithClfHeadAndAdapters(self.config["config"],
                                                           self.config["config_ft"]).to(self.device)
            state_dict = torch.load(cached_path(os.path.join(model_path, "model_weights.pth")),
                                    map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        except:
            raise Exception("Require a valid transformer model file ({0}/model_weights.pth) "
                            "and its config file ({0}/model_training_args.bin)."
                            .format(model_path))

    def encode(self, inputs):
        return list(self.tokenizer.convert_tokens_to_ids(o) for o in inputs)

    def score(self, text: str) -> int:
        "Return an integer value of predicted class from the transformer model."
        import torch
        import torch.nn.functional as F

        self.model.eval()   # Disable dropout
        clf_token = self.tokenizer.vocab['[CLS]']  # classifier token
        pad_token = self.tokenizer.vocab['[PAD]']  # pad token
        max_length = self.config['config'].num_max_positions  # Max length from trained model
        inputs = self.tokenizer.tokenize(text)
        if len(inputs) >= max_length:
            inputs = inputs[:max_length - 1]
        ids = self.encode(inputs) + [clf_token]

        with torch.no_grad():   # Disable backprop
            tensor = torch.tensor(ids, dtype=torch.long).to(self.device)
            tensor = tensor.reshape(1, -1)
            tensor_in = tensor.transpose(0, 1).contiguous()  # to shape [seq length, 1]
            logits = self.model(tensor_in,
                                clf_tokens_mask=(tensor_in == clf_token),
                                padding_mask=(tensor == pad_token))
        val, _ = torch.max(logits, 0)
        val = F.softmax(val, dim=0).detach().cpu().numpy()
        # To train the transformer in PyTorch we zero-indexed the labels.
        # Now we increment the predicted label by 1 to match with those from other classifiers.
        pred = int(val.argmax()) + 1
        return pred

    def predict(self, train_file: None, test_file: str, lower_case: bool) -> pd.DataFrame:
        "Use tqdm to display model prediction status bar"
        # pip install tqdm
        from tqdm import tqdm
        tqdm.pandas()
        df = self.read_data(test_file, lower_case)
        df['pred'] = df['text'].progress_apply(self.score)
        return df
