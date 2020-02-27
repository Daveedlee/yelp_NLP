import pandas as pd
import re
import string
import numpy as np
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
from textblob import TextBlob as tb
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from tqdm import tqdm
import langdetect
import pickle
import seaborn as sns


def overall_cleaner(df, list_of_columns):
    """
    takes a dataframe and list of columns and returns a new dataframe only with those columns
    df = pandas data frame
    list_of_columns = a list of columns
    """
    new_df = pd.DataFrame(None)
    for i in list_of_columns:
        new_df[i] = df.loc[:,i]
    return new_df


def lang_filter(df):
    df['lang'] = None
    for i in tqdm(range(len(df.text))):
        try:
            df['lang'].iloc[i] = langdetect.detect(df.text.iloc[i])
        except:
            df['lang'].iloc[i] = 'n/a'
    df = df[df['lang'] == 'en']
    return df


def clean_text_round(text):
    '''Make text lowercase, remove text in square brackets,
    remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


nlp = English()
nlp.max_length = 3000000
tokenizer = Tokenizer(nlp.vocab)


def wordcloud_auto(df):
    """
    Takes a df, and turns it into a word cloud, if possible.
    The data frame must have a column named 'text' in order for this function to
    run properly.
    """

    if 'text' in df.columns:
        df['gs_remove'] = df.text.apply(lambda x: remove_stopwords(x))
        df['nlp'] = df.gs_remove.apply(lambda x: nlp(x))
        lemma = []

        for i in iter(df.nlp):
            for j in i:
                lemma.append(j.lemma_)

        STOPWORDS.add('PRON')
        stopwords = STOPWORDS

        wordcloud = WordCloud(stopwords=stopwords,
                              background_color='White',
                              width=1000, height=500, max_words = 30).generate(' '.join(lemma))

        plt.figure(figsize=(24,16))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show();
    else:
        print("Cannot locate the text column. Please use pd.dataframe.rename() to specify the column.")


def s_and_p(df):
    """
    Takes a data frame and adds sentiment polarity and subjectivity to the existing data frame.
    """
    df['s_and_p'] = df.text.apply(lambda x: tb(x).sentiment)
    df['polarity'] = df['s_and_p'].apply(lambda x: x[0])
    df['subjectivity'] = df['s_and_p'].apply(lambda x: x[1])
    df.drop('s_and_p', axis=1, inplace=True)
    return df


class neural_modeling(object):
    def __init__(self, path):
        import pickle
        from sklearn.model_selection import train_test_split
        import pandas as pd
        from keras.preprocessing import sequence, text

        self.path = path
        self.df = pickle.load(open(self.path, 'rb'))
        self.text = self.df.text
        self.total_vocab = set(word for text in self.df.text for word in text.split(' '))

        self.tokenizer = text.Tokenizer(num_words=len(self.total_vocab))
        self.tokenizer.fit_on_texts(self.text)
        tokenized_list = self.tokenizer.texts_to_sequences(self.text)
        self.padded_seq = sequence.pad_sequences(tokenized_list, maxlen=150)
        self.target_dummies = pd.get_dummies(self.df.review_rating)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.padded_seq,
                                                                                pd.get_dummies(self.df.review_rating),
                                                                                test_size=0.2)


    def baseline(self):
        from keras.models import Sequential
        from keras.layers import Embedding, LSTM, Dense, GlobalMaxPool1D, Dropout, Bidirectional, Conv1D, Activation

        model = Sequential()
        embedding_size=150

        model.add(Embedding(len(self.total_vocab), embedding_size))
        model.add(Conv1D(64,
                         kernel_size=5,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPool1D())
        model.add(Dense(5, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', 'mse'])

        model.summary()
        return model


    def fit_evaluate(self, n_epoch=5, review_sample=None, save=False):

        model = self.baseline()
        model.fit(self.X_train, self.y_train, epochs=n_epoch, batch_size=100, validation_split=0.1)

        print('-------Accuracy-------', '\n')
        print(model.evaluate(self.X_test, self.y_test))
        print(model.model.metrics_names)

        if review_sample:
            from keras.preprocessing import sequence

            tokenized_s = self.tokenizer.texts_to_sequences(review_sample)
            rating_pred = np.argmax(model.predict(sequence.pad_sequences(tokenized_s, maxlen=150))[0])
            print('That review looks like', (rating_pred + 1), 'star(s)!')

        if save:
            model.save('models/' + self.path + '.h5')

    def cross_val(self, n_splits=3):
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import KFold, cross_val_score

        estimator = KerasClassifier(build_fn=self.baseline, epochs=3, batch_size=128)
        Kfold = KFold(n_splits=n_splits, shuffle=True)
        results = cross_val_score(estimator, self.X_train, self.y_train, cv=Kfold)
        print(np.mean(results))


class non_neural(object):
    def __init__(self, path):
        import pickle
        import pandas as pd
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split

        self.df = pickle.load(open(path, 'rb'))
        self.word_list = list(self.df.text)
        self.vectorizer = CountVectorizer()
        vectorized = self.vectorizer.fit_transform(self.word_list)
        self.matrix_vec = pd.DataFrame(vectorized.toarray(), columns=self.vectorizer.get_feature_names())

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.matrix_vec,
                                                                                self.df.review_rating,
                                                                                test_size=0.33)
    def rf(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, roc_auc_score

        rf = RandomForestClassifier(n_estimators=64, class_weight = 'balanced', verbose=True, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)

        y_hat_train = rf.predict(self.X_train)
        y_hat_test = rf.predict(self.X_test)

        acc_random_forest = round(rf.score(self.X_test, self.y_test) * 100, 2)
        print('Model Accuracy: ', acc_random_forest)

        print('Train', classification_report(self.y_train, y_hat_train),
              'Test', classification_report(self.y_test, y_hat_test),
              sep='\n-------------------------------------------------------\n')

        y_score = rf.predict_proba(self.X_test)
        print('ovo', roc_auc_score(self.y_test, y_score, multi_class='ovo'),
              'ovr', roc_auc_score(self.y_test, y_score, multi_class='ovr'),
              sep='\n-------------------------------------------------------\n')


    def bnb(self):
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.metrics import classification_report, roc_auc_score

        bnb = BernoulliNB()
        bnb.fit(self.X_train, self.y_train)

        y_hat_train = bnb.predict(self.X_train)
        y_hat_test = bnb.predict(self.X_test)

        acc_bnb = round(bnb.score(self.X_test, self.y_test) * 100, 2)
        print('Model Accuracy: ',acc_bnb)

        print('Naive Bayes:\n 1. train 2. test')
        print(classification_report(self.y_train, y_hat_train), classification_report(self.y_test, y_hat_test),
              sep='\n-------------------------------------------------------\n')

        y_score = bnb.predict_proba(self.X_test)
        print('ovo', roc_auc_score(self.y_test, y_score, multi_class='ovo'),
            'ovr', roc_auc_score(self.y_test, y_score, multi_class='ovr'),
              sep='\n-------------------------------------------------------\n')


class simple_bar():
    def __init__(self, path, gram=1, n_most=20):
        self.df = pickle.load(open(path, 'rb'))
        self.ngrams = []
        self.index = []
        self.value = []

        for sentence in self.df.text:
            sentence = remove_stopwords(sentence)
            splitted = sentence.split(' ')

            for element in splitted:
                if element == '':
                    splitted.remove(element)

            while len(splitted) > (gram-1):
                    self.ngrams.append(tuple(splitted[0:gram]))
                    splitted.pop(0)

        self.count = Counter(self.ngrams).most_common(n_most)

        for i in self.count:
            if len(self.count[0][0]) == 2:
                self.index.append('\n'.join([i[0][0], i[0][1]]))
                self.value.append(i[1])
            elif len(self.count[0][0]) == 3:
                self.index.append('\n'.join([i[0][0], i[0][1], i[0][2]]))
                self.value.append(i[1])
            else:
                print('Neither 2 nor 3')

    def barplot(self, size=(35,13)):
        plt.figure(figsize=size)
        barplot = sns.barplot(self.index, self.value)
        barplot.tick_params(labelsize=25)
