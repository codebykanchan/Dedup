import pandas as pd
import re
import spacy
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import gensim
from gensim import corpora
from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
from gensim.parsing import preprocess_string

regex = r'([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#.]?[\w-]+)*\/?'
remove_urls = lambda s: re.sub(regex, '', s)

convert_to_lower = lambda s: s.lower()

remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)

CLEAN_FILTERS = [
                remove_urls,
                strip_tags,
                strip_numeric,
                strip_punctuation, 
                strip_multiple_whitespaces, 
                convert_to_lower,
                remove_stopwords,
                remove_single_char]
 
 
def gensim_clean_pipeline(document):
    '''
    preprocessing of string data
    '''
    processed_words = preprocess_string(document, CLEAN_FILTERS)
    return processed_words


stop_words = stopwords.words('english')
def remove_stopwords(text):
    '''
    Remove Stopwords
    '''
    output = ' '.join([i for i in text if i not in stop_words])
    return(output)


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def lemmatization(texts,allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]): 
    '''
    Lemmatize the text
    '''
    doc = nlp(texts) 
    output = ' '.join([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
    return(output) 
    

def clean_pipeline(df, columns_lst):
    '''
    Text cleanup
    '''
    col_new = []
    for col in columns_lst:
        df[col+'_new'] = df[col].apply(gensim_clean_pipeline).apply(remove_stopwords).apply(lemmatization)
        col_new.append(col+'_new')
    return(col_new)    

