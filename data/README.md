# Stanford Sentiment Treebank Dataset
The dataset used for this analysis is from the [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/code.html).

We are interested in the 5-class version of this dataset in the PTB tree format, which can be downloaded from [this link](https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip).

To convert the tree-structured data to a tabular form that we can work with during text classification, we use the [pytreebank](https://pypi.org/project/pytreebank/) library (install using `pip`).

```
# Load data
import pytreebank
dataset = pytreebank.load_sst('./sst')
# Store train, dev and test in separate files
for category in ['train', 'test', 'dev']:
    with open('../data/sst/sst_{}.txt'.format(category), 'w') as outfile:
        for item in dataset[category]:
            outfile.write("__label__{}\t{}\n".format(
                item.to_labeled_lines()[0][0] + 1,
                item.to_labeled_lines()[0][1]
            ))
# Print the length of the training set
print(len(dataset['train']))
```

Note that this will write the class labels with the prefix `__label__` for each class. This is because classifiers like fastText and Flair NLP require the class data labels in this format. For other classifiers, we simply strip this prefix before training the model. 