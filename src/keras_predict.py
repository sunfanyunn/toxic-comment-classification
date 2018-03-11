from fastText import load_model
##################
# parameters
#################
window_length = 200 # The amount of words we look at per example. Experiment with this.
n_features = 300
batch_size=32

classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]
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

def build_model(optimizer):
    inp = Input(shape=(window_length, n_features ))
#    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(inp)
    x = Bidirectional(GRU(320, return_sequences=True))(x)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(GRU(320, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
#    print(max_pool.shape)
#    def https://github.com/Wronskia/Sentiment-Analysis-on-Twitter-data.gitslice(x):
#        return x[:,-1,:]
    conc = concatenate([avg_pool, max_pool])
    x = Dropout(0.5)(conc)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()
    return model

if __name__=='__main__':
    ft_model =load_model('ft_model.bin')
    print('Load ft_model')
    from keras.models import load_model
    import sys
    model = load_model(sys.argv[1])
    print('Load keras model')
    test = pd.read_csv('../input/test_clean.csv')
    x_test = df_to_data(test)
    print('After transforming testing data')
    y_pred = model.predict(x_test, batch_size=batch_size)
    submission = pd.read_csv('../input/sample_submission.csv')
    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
    submission.to_csv('keras.csv', index=False)
    print('Done')
