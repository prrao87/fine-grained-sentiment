# Load data
import pytreebank
dataset = pytreebank.load_sst('./')
# Store train, dev and test in separate files
for category in ['train', 'test', 'dev']:
    with open('sst_{}.txt'.format(category), 'w') as outfile:
        for item in dataset[category]:
            outfile.write("__label__{}\t{}\n".format(
                item.to_labeled_lines()[0][0] + 1,
                item.to_labeled_lines()[0][1]
            ))
# Print the length of the training set
print(len(dataset['train']))
