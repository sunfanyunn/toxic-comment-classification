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

#class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_names = [ 'obscene', 'severe_toxic', 'threat', 'insult' , 'identity_hate','toxic']


train = pd.read_csv('../input/internal_train.csv').fillna(' ')
valid = pd.read_csv('../input/internal_test_with_answer.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
print('Loaded')

train_text = train['comment_text']
valid_text = valid['comment_text']
test_text = test['comment_text']

from sklearn.feature_extraction.text import CountVectorizer

word_vectorizer = CountVectorizer(
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_features=50000)

train_word_features = word_vectorizer.fit_transform(train_text)
print('Word Count 1/2')
valid_word_features = word_vectorizer.transform(valid_text)
test_word_features = word_vectorizer.transform(test_text)
print('Word Count 2/2')

word_vectorizer = CountVectorizer(
    strip_accents='unicode',
    analyzer='char_wb',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_features=50000)

train_charwb_features = word_vectorizer.fit_transform(train_text)
print('charwb Count 1/2')
valid_charwb_features = word_vectorizer.transform(valid_text)
test_charwb_features = word_vectorizer.transform(test_text)
print('charwb Count 2/2')

char_vectorizer = CountVectorizer(
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
train_char_features = char_vectorizer.fit_transform(train_text)
print('Char Count 1/2')
valid_char_features = char_vectorizer.transform(valid_text)
test_char_features = char_vectorizer.transform(test_text)
print('Char Count 2/2')


# In[ ]:


train_features = hstack([ train_charwb_features, train_char_features, train_word_features])
print('HStack 1/3')
valid_features = hstack([ valid_charwb_features, valid_char_features, valid_word_features])
print('HStack 2/3')
test_features = hstack([ test_charwb_features, test_char_features, test_word_features])
print('HStack 3/3')


# In[ ]:


submission = pd.DataFrame.from_dict({'id': test['id']})

train.drop('comment_text', axis=1, inplace=True)
valid.drop('comment_text', axis=1, inplace=True)

del test

del train_text
del test_text
del valid_text

del train_char_features
del valid_char_features
del test_char_features

del train_word_features
del valid_word_features
del test_word_features

del train_charwb_features
del valid_charwb_features
del test_charwb_features
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

train_append = None
valid_append = None
test_append = None

for class_name in class_names:
    print(class_name)
    train_target = train[class_name]
    valid_target = valid[class_name]

    model = LogisticRegression(solver='sag')
    sfm = SelectFromModel(model)

    print('Train: before feature selection\'s shape: ', train_features.shape)
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    print('After', train_sparse_matrix.shape)

    print('Valid: before feature selection\'s shape: ', valid_features.shape)
    valid_sparse_matrix = sfm.fit_transform(valid_features, valid_target)
    print('After', valid_sparse_matrix.shape)

    print('Test: before feature selection\'s shape:', test_features.shape)
    test_sparse_matrix = sfm.transform(test_features)
    print('After', test_sparse_matrix.shape)

    train_sparse_matrix = train_sparse_matrix.astype(np.float32)
    valid_sparse_matrix = valid_sparse_matrix.astype(np.float32)
    test_sparse_matrix = test_sparse_matrix.astype(np.float32)

    print(train_sparse_matrix.shape)
    if train_append is not None:
        train_sparse_matrix = hstack([train_sparse_matrix, train_append]).astype(np.float32)
    print(train_sparse_matrix.shape)

    print(valid_sparse_matrix.shape)
    if valid_append is not None:
        valid_sparse_matrix = hstack([valid_sparse_matrix, valid_append]).astype(np.float32)
    print(valid_sparse_matrix.shape)

    print(test_sparse_matrix.shape)
    if test_append is not None:
        test_sparse_matrix = hstack([test_sparse_matrix, test_append]).astype(np.float32)
    print(test_sparse_matrix.shape)

    d_train = lgb.Dataset(train_sparse_matrix, label=train[class_name].astype(np.float32))
    d_valid = lgb.Dataset(valid_sparse_matrix, label=valid[class_name].astype(np.float32))
    watchlist = [d_train, d_valid]

    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=rounds_lookup[class_name],
                      valid_sets=watchlist,
                      verbose_eval=10)
    submission[class_name] = model.predict(test_sparse_matrix)

    if train_append is None:
        train_append = np.hstack([np.expand_dims(model.predict(train_sparse_matrix), axis=1)])
        valid_append = np.hstack([np.expand_dims(model.predict(valid_sparse_matrix), axis=1)])
        test_append = np.hstack([np.expand_dims(submission[class_name], axis=1)])
    else:
        train_append = np.hstack([train_append, np.expand_dims(model.predict(train_sparse_matrix), axis=1)])
        valid_append = np.hstack([valid_append, np.expand_dims(model.predict(valid_sparse_matrix), axis=1)])
        test_append = np.hstack([test_append,   np.expand_dims(submission[class_name], axis=1)])

    print(train_append.shape)
    print(valid_append.shape)
    print(test_append.shape)

    print(final_val_score)
    print('final score:', np.mean(final_val_score))
submission.to_csv('countvectorizer.csv', index=False)
