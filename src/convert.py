import pandas as pd
from tqdm import tqdm

train = pd.read_csv('../input/train.csv')
internal_train = pd.read_csv('../input/internal_train.csv')
internal_test = pd.read_csv('../input/internal_test.csv')

dic = { c:idx for idx, c in enumerate(list(train['comment_text'])) }
print(len(dic))

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for c in class_names: internal_test[c] = 0

for idx, row in tqdm(internal_test.iterrows()):
    txt = row[1]
    trainidx = dic[txt]
    for c in class_names: internal_test[c][idx] = train[c][idx]


print(internal_test.head())

internal_test.to_csv('internal_test_with_answer.csv')

