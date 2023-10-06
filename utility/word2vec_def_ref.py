#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import gensim
import struct
import binascii
import sys
import nltk
import bisect
import os
from numba import jit 
import time

from nltk.tokenize import word_tokenize


# In[2]:


DATA_DIR = '../data/'
INPUT_DATA_FILE = 'angularjs_processed_withimagetext.csv'
OUTPUT_DATA_FILE = 'angularjs_w2v.csv'
MODEL_DIR = '../models/'
MODEL_FILES = ['glove-wiki-gigaword-300.bin','word2vec-google-news-300.bin','fasttext-wiki-news-subwords-300.bin']
W2V_COLS = ['W2V_glove','W2V_google','W2V_fasttext']
VECTOR_SIZE = 300

INPUT_DATA_DIR = '../data/'
INPUT_DATA_FILE = 'angularjs_w2v.csv'

KEYS = ['W2V_glove','W2V_google','W2V_fasttext']
MAX_SIMILAR = 5
VECTOR_SIZE = 300


# In[4]:


def sent2vec(model,s):

    words = str(s).lower()
    words = word_tokenize(words)
#     words = [w for w in words if not w in stop_words]
#     words = [w for w in words if w.isalpha()]

    featureVec = np.zeros((300,), dtype="float32")
#     print(featureVec[0])
    nwords = 0

    for w in words:
        try:
            featureVec = np.add(featureVec, model[w])
            nwords = nwords + 1
        except:
            continue

        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
    return featureVec


# In[5]:


def calcVectors(df,W2V_COLS,dir=MODEL_DIR,models=MODEL_FILES):
    db = pd.DataFrame()
    db['Number'] = df['Number']
    db[W2V_COLS] = None
    
    for i in range(len(models)):
        db[W2V_COLS[i]] = None
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(dir + models[i] ,binary=True)

        for indx in df.index:
            vec = sent2vec(word2vec,str(df['IssueText'][indx]))
            binval = struct.pack('f'*VECTOR_SIZE, *vec)
            valStr = binascii.b2a_base64(binval, newline=False).decode('ascii')
            db[W2V_COLS[i]][indx] = valStr
    return(db) 


# In[8]:


@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


# In[9]:


def cossim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))


# In[4]:


#@jit(nopython=True)
def find_similar_doc(column,db,MAX_SIMILAR):
    cols = []
    for i in range(MAX_SIMILAR):
        cols = cols + [column + '_' + str(i)] + [column + '_score' + str(i)]
    df = pd.DataFrame()
    df['Number'] = db['Number']
    df[cols] = None
    vectdict = {}
    print('Computing Vectors')
    
    zeroVec = list(np.zeros((VECTOR_SIZE,), dtype="float32"))
    for indx in db.index:
        binval = binascii.a2b_base64(db[column][indx])
        val = np.array(struct.unpack('f'*VECTOR_SIZE, binval))
        if list(val) == zeroVec:
            continue
        vectdict[df['Number'][indx]] = val
    dictkeys = list(vectdict.keys())
    
    for indx in db.index:
        curNum = db['Number'][indx]
        if curNum not in dictkeys:
            continue
        curVec = vectdict[curNum]
            
        simScore = [-2.0]*MAX_SIMILAR
        simNumber = [0]*MAX_SIMILAR
        if indx%500 == 0:
            print(f'Done till index {indx}')
    
        for key in dictkeys:
            if key == curNum:
                continue
            currScore = cosine_similarity_numba(curVec, vectdict[key])
            
            if [currScore] > simScore:
                idx = bisect.bisect(simScore, currScore)
                simScore.insert(idx, currScore)
                simNumber.insert(idx, key)
                simScore = simScore[1:]
                simNumber = simNumber[1:]

        for i in range(MAX_SIMILAR):
            df[column + '_' + str(i)][indx] = simNumber[i]
            df[column + '_score' + str(i)][indx] = simScore[i]
            
    df.to_csv(INPUT_DATA_DIR + column + '.csv', index=False, encoding='utf-8') 
    return(df)


# In[ ]:




