import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
tqdm.pandas()

def print_accuracy(df, pred_column):
    "Print f1 score and accuracy after making predictions"
    f1_macro = f1_score(df['truth'], df[pred_column], average='macro')
    acc = accuracy_score(df['truth'], df[pred_column])*100
    return f1_macro, acc

def flair_score(text):
    doc = Sentence(text)
    flair_model.predict(doc)
    pred = int(doc.labels[0].value)
    return pred

# Read test data
df = pd.read_csv('data/sst/sst_test.txt', sep='\t', header=None,
                   names=['truth', 'sentence'],
                  )
df['truth'] = df['truth'].str.replace('__label__', '')
df['truth'] = df['truth'].astype(int).astype('category')

flair_model = TextClassifier.load("models/flair/best-model.pt")

df['flair_pred'] = df['sentence'].progress_apply(flair_score)
# Get model accuracy and f1 score
acc = print_accuracy(df, 'flair_pred')
print("Macro F1-score: {}\nAccuracy: {}".format(acc[0], acc[1]))
