"""
This code is adapted from the original version in this excellent notebook:
https://github.com/ben0it8/containerized-transformer-finetuning/blob/develop/research/finetune-transformer-on-imdb5k.ipynb
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm
from typing import Tuple

from pytorch_transformers import BertTokenizer

TEXT_COL, LABEL_COL = 'text', 'truth'
MAX_LENGTH = 256
DATASET_DIR = "../../data/sst"
n_cpu = multiprocessing.cpu_count()


def read_sst5(data_dir, colnames=[LABEL_COL, TEXT_COL]):
    datasets = {}
    for t in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"sst_{t}.txt"), sep='\t', header=None, names=colnames)
        df[LABEL_COL] = df[LABEL_COL].str.replace('__label__', '')
        df[LABEL_COL] = df[LABEL_COL].astype(int)   # Categorical data type for truth labels
        df[LABEL_COL] = df[LABEL_COL] - 1  # Zero-index labels for PyTorch
        datasets[t] = df
    return datasets


class TextProcessor:
    def __init__(self, tokenizer, label2id: dict, clf_token, pad_token, max_length):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.clf_token = clf_token
        self.pad_token = pad_token

    def encode(self, input):
        return list(self.tokenizer.convert_tokens_to_ids(o) for o in input)

    def token2id(self, item: Tuple[str, str]):
        "Convert text (item[0]) to sequence of IDs and label (item[1]) to integer"
        assert len(item) == 2   # Need a row of text AND labels
        label, text = item[0], item[1]
        assert isinstance(text, str)   # Need position 1 of input to be of type(str)
        inputs = self.tokenizer.tokenize(text)
        # Trim or pad dataset
        if len(inputs) >= self.max_length:
            inputs = inputs[:self.max_length - 1]
            ids = self.encode(inputs) + [self.clf_token]
        else:
            pad = [self.pad_token] * (self.max_length - len(inputs) - 1)
            ids = self.encode(inputs) + [self.clf_token] + pad

        return np.array(ids, dtype='int64'), self.label2id[label]

    def process_row(self, row):
        "Calls the token2id method of the text processor for passing items to executor"
        return self.token2id((row[1][LABEL_COL], row[1][TEXT_COL]))

    def create_dataloader(self,
                          df: pd.DataFrame,
                          batch_size: int = 32,
                          shuffle: bool = False,
                          valid_pct: float = None):
        "Process rows in pd.DataFrame using n_cpus and return a DataLoader"

        tqdm.pandas()
        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            result = list(
                tqdm(executor.map(self.process_row, df.iterrows(), chunksize=8192),
                     desc=f"Processing {len(df)} examples on {n_cpu} cores",
                     total=len(df)))

        features = [r[0] for r in result]
        labels = [r[1] for r in result]

        dataset = TensorDataset(torch.tensor(features, dtype=torch.long),
                                torch.tensor(labels, dtype=torch.long))

        if valid_pct is not None:
            valid_size = int(valid_pct * len(df))
            train_size = len(df) - valid_size
            valid_dataset, train_dataset = random_split(dataset, [valid_size, train_size])
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            return train_loader, valid_loader

        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=0,
                                 shuffle=shuffle,
                                 pin_memory=torch.cuda.is_available())
        return data_loader


if __name__ == "__main__":
    datasets = read_sst5(DATASET_DIR)
    print(datasets['train'].head(10))
    labels = list(set(datasets["train"][LABEL_COL].tolist()))
    label2int = {label: i for i, label in enumerate(labels)}
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    clf_token = tokenizer.vocab['[CLS]']  # classifier token
    pad_token = tokenizer.vocab['[PAD]']  # pad token
    proc = TextProcessor(tokenizer, label2int, clf_token, pad_token, max_length=MAX_LENGTH)

    train_dl = proc.create_dataloader(datasets["train"], batch_size=32)
    valid_dl = proc.create_dataloader(datasets["dev"], batch_size=32)
    test_dl = proc.create_dataloader(datasets["test"], batch_size=32)

    print(len(train_dl), len(valid_dl), len(test_dl))
