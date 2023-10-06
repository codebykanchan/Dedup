#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re

#display max column width
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth', None)

#spacy packages
import spacy

# nltk packages
import nltk
from nltk.corpus import stopwords

# Gensim packages
import gensim
from gensim import corpora
from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
from gensim.parsing import preprocess_string


# In[2]:


# Custom filters for Text Processing for gensim

regex = r'([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#.]?[\w-]+)*\/?'
remove_urls = lambda s: re.sub(regex, '', s)

convert_to_lower = lambda s: s.lower()

remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)

# Filters to be executed in pipeline
CLEAN_FILTERS = [
                remove_urls,
                strip_tags,
                strip_numeric,
                strip_punctuation, 
                strip_multiple_whitespaces, 
                convert_to_lower,
                remove_stopwords,
                remove_single_char]


# In[3]:


# Method does the filtering of all the unrelevant text elements using gensim 
def gensim_clean_pipeline(document):
    # Invoking gensim.parsing.preprocess_string method with set of filters
    processed_words = preprocess_string(document, CLEAN_FILTERS)
    
    return processed_words


# In[4]:


# nltk stop words
stop_words = stopwords.words('english')
def remove_stopwords(text):
    output = ' '.join([i for i in text if i not in stop_words])
    return(output)


# lemitization using spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts,allowed_postags=["NOUN", "PROPN","ADJ", "VERB", "ADV"]): 

    doc = nlp(texts) 
    output = ' '.join([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
    return(output)

#lemitization using nltk
from nltk.stem import WordNetLemmatizer 

def lemmit(texts):
    lemmatizer = WordNetLemmatizer()
    return([lemmatizer.lemmatize(w) for w in texts])   


# In[7]:


def clean_pipeline(df, columns_lst):
    col_new = []
    for col in columns_lst:
        df[col+'_new'] = df[col].apply(gensim_clean_pipeline).apply(remove_stopwords).apply(lemmatization)
        col_new.append(col+'_new')
    return(col_new)    

def count_nan_values(df, cols,filler):
    # before filling
    for col in cols:
        print(f'{col} has {df[col].isna().sum()} NAN values')
    print('\n ------- \n')
    df[cols] = df[cols].fillna('').astype(str)   
    for col in cols:
        print(f'{col} has {df[col].isna().sum()} NAN values')  

