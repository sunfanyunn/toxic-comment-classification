
# coding: utf-8

# In[40]:


# The goal of this kernel is to demonstrate that LightGBM can have predictive
# performance in line with that of a logistic regression. The theory is that
# labeling is being driven by a few keywords that can be picked up by trees.
#
# With some careful tuning, patience with runtimes, and additional feature
# engineering, this kernel can be tuned to slightly exceed the best
# logistic regression. Best of all, the two approaches (LR and LGB) blend
# well together.
#
# Hopefully, with some work, this could be a good addition to your ensemble.

import gc
import pandas as pd

from scipy.sparse import csr_matrix, hstack


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb



# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/internal_train.csv').fillna(' ')
test = pd.read_csv('../input/internal_test.csv').fillna(' ')
print('Loaded')

train_text = train['comment_text']
test_text = test['comment_text']

# In[ ]:


from sklearn.feature_extraction.text import HashingVectorizer

word_vectorizer = HashingVectorizer(
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2))

train_word_features = word_vectorizer.fit_transform(train_text)
print('Word Count 1/2')
test_word_features = word_vectorizer.transform(test_text)
print('Word Count 2/2')

word_vectorizer = HashingVectorizer(
    strip_accents='unicode',
    analyzer='char_wb',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2))

train_charwb_features = word_vectorizer.fit_transform(train_text)
print('charwb Count 1/2')
test_charwb_features = word_vectorizer.transform(test_text)
print('charwb Count 2/2')

char_vectorizer = HashingVectorizer(
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6))
train_char_features = char_vectorizer.fit_transform(train_text)
print('Char Count 1/2')
test_char_features = char_vectorizer.transform(test_text)
print('Char Count 2/2')


# In[ ]:


train_features = hstack([ train_charwb_features, train_char_features, train_word_features])
print('HStack 1/2')
test_features = hstack([ test_charwb_features, test_char_features, test_word_features])
print('HStack 2/2')


# In[ ]:


submission = pd.DataFrame.from_dict({'id': test['id']})

train.drop('comment_text', axis=1, inplace=True)
del test
del train_text
del test_text
del train_char_features
del test_char_features
del train_word_features
del test_word_features
gc.collect()


# In[ ]:


import numpy as np
params = {'learning_rate': 0.2,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 2,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.6,
              'nthread': 4,
              'lambda_l1': 1,
              'lambda_l2': 1}

rounds_lookup = {'toxic': 140,
                 'severe_toxic': 50,
                 'obscene': 80,
                 'threat': 80,
                 'insult': 70,
                 'identity_hate': 80}

final_val_score = []
for class_name in class_names:
    print(class_name)
    train_target = train[class_name]
    model = LogisticRegression(solver='sag')
    sfm = SelectFromModel(model, threshold=.2)
    
    print('Train: before feature selection\'s shape: ', train_features.shape)
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    print('After', train_sparse_matrix.shape)
    #train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target, test_size=0.05, random_state=144)
    print('Test: before feature selection\'s shape:', test_features.shape)
    test_sparse_matrix = sfm.transform(test_features)
    print('After', test_sparse_matrix.shape)

    train_sparse_matrix = train_sparse_matrix.astype(np.float32)
    test_sparse_matrix = test_sparse_matrix.astype(np.float32)
    
    y_train = train[class_name]
    d_train = lgb.Dataset(train_sparse_matrix, label=y_train.astype(np.float32))
    
    res = lgb.cv(params,
                 train_set=d_train,
                 num_boost_round=rounds_lookup[class_name],
                 verbose_eval=10)
    
    rounds = np.argmax(res['auc-mean'])
    final_val_score.append(res['auc-mean'][rounds])
    print(rounds)
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds,
                      verbose_eval=10)
    
    submission[class_name] = model.predict(test_sparse_matrix)

    print(final_val_score)
    print('final score:', np.mean(final_val_score))
submission.to_csv('internal_hashingvectorizer.csv', index=False)

