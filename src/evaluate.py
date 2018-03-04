from sklearn.metrics import roc_auc_score
import pandas as pd
import sys

test_pred = pd.read_csv(sys.argv[1])
test_ans = pd.read_csv('../input/internal_test_with_answer.csv')
print(test_ans.columns)
print(test_ans.head())
print(len(test_ans))
print(len(test_pred))

auc = []

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for c in class_names:
    sc = roc_auc_score(test_ans[c], test_pred[c])
    print(c, sc)
    auc.append(sc)

print(auc)
import numpy as np
print(np.mean(auc))
