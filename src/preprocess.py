import re
from nltk.corpus import wordnet
#import enchant
import itertools

"""
Below, we used three normalizazion dictionaries from those links :
http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/
http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz
http://luululu.com/tweet/typo-corpus-r1.txt

These dictionaries have been built by researchers from the noisy tweets and corrects
the common mistakes and abbreviations that are made in the english
They help us clean the noise.

"""
dico = {}
dico1 = open('../data/dicos/dico1.txt', 'rb')
for word in dico1:
    word = word.decode('utf8')
    word = word.split()
    dico[word[1]] = word[3]
dico1.close()
dico2 = open('../data/dicos/dico2.txt', 'rb')
for word in dico2:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico2.close()
dico3 = open('../data/dicos/dico2.txt', 'rb')
for word in dico3:
    word = word.decode('utf8')
    word = word.split()
    dico[word[0]] = word[1]
dico3.close()

#d = enchant.Dict('en_US')


def remove_repetitions(tweet):
    """
    Functions that remove noisy character repetition like for instance :
    llloooooooovvvvvve ====> love
    This function reduce the number of character repetition to 2 and checks if the word belong the english
    vocabulary by use of pyEnchant and reduce the number of character repetition to 1 otherwise
    Arguments: tweet (the tweet)
    """

    tweet=tweet.split()
    for i in range(len(tweet)):
        tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i])).replace('#', '')
        if len(tweet[i])>0:
            if not wordnet.synsets(tweet[i]):
                tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet=' '.join(tweet)
    return tweet

def correct_spell(tweet):
    """
    Function that uses the three dictionaries that we described above and replace noisy words

    Arguments: tweet (the tweet)

    """
    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in dico.keys():
            tweet[i] = dico[tweet[i]]
    tweet = ' '.join(tweet)
    return tweet

def clean(tweet):
    """
    Function that cleans the tweet using the functions above and some regular expressions
    to reduce the noise

    Arguments: tweet (the tweet)

    """
    #Separates the contractions and the punctuation
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)
    tweet = remove_repetitions(tweet)
    tweet = correct_spell(tweet)
    return tweet.strip().lower()

if __name__ =='__main__':
    import pandas as pd
    from tqdm import tqdm
    test = pd.read_csv('../input/test.csv')
    txts = test['comment_text']
    for i, txt in tqdm(enumerate(txts)):
#        print('=======================================')
#        print(txt)
#        print('=======================================')
#        print(clean(txt))
#        print('=======================================')
#        input()
        test['comment_text'][i] = clean(txt)

    test.to_csv('test_clean.csv', index=False)
