import pandas as pd
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#tfidf = pd.read_csv('../input/internal_tfidfvectorizer.csv')
#count = pd.read_csv('../input/internal_countvectorizer.csv')
#hashing = pd.read_csv('../input/internal_hashingvectorizer.csv')
#a = pd.read_csv('../input/bi-gru-cnn-poolings.csv')
#print(a.shape)
#b = pd.read_csv('../input/ensemble.csv')
#print(b.shape)
#c = pd.read_csv('../input/bi-lstm-cnn.csv')
#print(c.shape)



a = pd.read_csv('../input/ensemble_again.csv')
print(a.shape)
b = pd.read_csv('../input/hight_of_blend_v2.csv')
print(b.shape)
c = pd.read_csv('../input/corr_blend.csv')
print(c.shape)


ans = pd.read_csv('../input/sample_submission.csv')
print(ans.shape)
for d in class_names:
    ans[d] = a[d] + b[d] + c[d]
ans.to_csv('ensemble_final.csv', index=False)
