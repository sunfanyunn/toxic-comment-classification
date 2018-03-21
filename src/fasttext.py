# Use scikit-learn to grid search the batch size and epochs
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_validation import StratifiedKFold
from keras.models import Sequential, load_model

from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Use scikit-learn to grid search the batch size and epochs
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Flatten, Lambda, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'
import re
import numpy as np
np.random.seed(42)
import pandas as pd
from fastText import load_model
from tqdm import tqdm

window_length = 200 # The amount of words we look at per example. Experiment with this.

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

print('\nLoading data')
train = pd.read_csv('../input/train_clean.csv')
test = pd.read_csv('../input/test_clean.csv')
train['comment_text'] = train['comment_text'].fillna('_empty_')
test['comment_text'] = test['comment_text'].fillna('_empty_')

classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

def build_model(optimizer):
    n_features = 300
    inp = Input(shape=(window_length, n_features ))
#    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(inp)
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
#    def https://github.com/Wronskia/Sentiment-Analysis-on-Twitter-data.gitslice(x):
#        return x[:,-1,:]
    x = concatenate([avg_pool, max_pool])
    x = Dropout(0.5)(x)
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()
    return model

print('\nLoading FT model')
ft_model =load_model('ft_model.bin')
#ft_model.
n_features = ft_model.get_dimension()
print(n_features)
#n_features = 300

def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = normalize(text)
    words = text.split()
    window = words[-window_length:]

    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in tqdm(enumerate(df['comment_text'].values)):
        x[i, :] = text_to_vector(comment)

    return x

def data_generator(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data

    while True: # Loop forever
        df = df.sample(frac=1) # Shuffle df each epoch

        for i, row in df.iterrows():
            comment = row['comment_text']

            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')

            batch_x[batch_i] = text_to_vector(comment)
            batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0


from customized_callback import MyModelCheckpoint

if __name__=='__main__':
    n_folds = 10
    epochs = 8
    ## tuning parameters
    batch_sizes = [32, 128, 512, 1024]
    optimizers = ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # Ready to start training:
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=233).split(train)
    for optimizer, batch_size in zip(optimizers, batch_sizes):
        print(optimizer, batch_size)

        for i, (trainidx, testidx) in enumerate(skf):
            if i >= 2: break
            print ("Running Fold", i+1, "/", n_folds)

            df_train = train.iloc[trainidx]
            df_val = train.iloc[testidx]
            print(df_train.shape)
            print(df_val.shape)

#            x_train = df_to_data(df_train)
#            y_train = train[classes].values
            x_val = df_to_data(df_val)
            y_val = df_val[classes].values

            model = None # Clearing the NN.
            model = build_model(optimizer)

            filepath = 'model_{}_{}.h5'.format(batch_size, optimizer)

            RocAuc =  MyModelCheckpoint(
                    filepath=filepath,
                    validation_data=(x_val, y_val),
                    monitor='roc_auc_score',
                    save_best_only=True,
                    verbose=1)

            training_steps_per_epoch = round(len(df_train) / batch_size)
            training_generator = data_generator(df_train, batch_size)
            model.fit_generator(
                training_generator,
                steps_per_epoch=training_steps_per_epoch,
                epochs=epochs,
                callbacks=[RocAuc],
            )
            model = load_model(filepath)
            print(auc_roc_score(y_val, model.predict(x_val)))

    x_test = df_to_data(test)
#y_test = test[classes].values
    submission = pd.read_csv('../input/sample_submission.csv')
    y_pred = model.predict(x_test, batch_size=batch_size)
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
    submission.to_csv('fastext_submission2.csv', index=False)
