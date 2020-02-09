import pandas as pd
import re
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords
from textblob import TextBlob as tb
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


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
        cloud.to_file('w_clouds/nevada.png')
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