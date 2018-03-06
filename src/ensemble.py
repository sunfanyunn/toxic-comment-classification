import pandas as pd
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
tfidf = pd.read_csv('../input/internal_tfidfvectorizer.csv')
count = pd.read_csv('../input/internal_countvectorizer.csv')
hashing = pd.read_csv('../input/internal_hashingvectorizer.csv')

ans = pd.read_csv('../input/internal_sample_submission.csv')

for c in class_names:
    if c == 'severe_toxic' or c == 'identity_hate':
        ans[c] = tfidf[c] + 2*count[c] + hashing[c]
    else:
        ans[c] = tfidf[c] + count[c] + hashing[c]

