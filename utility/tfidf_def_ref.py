#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from numba import jit
import bisect
import struct
import binascii
import os
from multiprocessing import Pool


# In[59]:


TFIDF_DATA_DIR = '../data/model_perf/'
TFIDF_INPUT_FILE = 'angularjs_processed_withimagetext.csv'
TFIDF_OUTPUT_FILE = 'tfidf_vec.csv'
TFIDF_OUTPUT_DATA_FILE ='tfidf_cos_sim.csv'
TFIDF_MAX_SIMILAR = 5
TFIDF_VECTOR_SIZE = 300
TFIDF_MODEL = 'tfidf_vector'


# In[9]:


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


# In[4]:


def calculate_tfidf_vectors(df,model,VECTOR_SIZE):
    global vectors
    global dictkeys
    global vectdict
    
    db = pd.DataFrame()
    db['Number'] = df['Number']
    db[model] = None
    
    vectorizer = TfidfVectorizer(max_features=VECTOR_SIZE)
    matrix = vectorizer.fit_transform(df['IssueText'].values.astype(str))

    vectors = matrix.asformat('array')
    numitems = vectors.shape[0]
    zerovec = np.array([0.0]*vectors.shape[1])

    print('sanitizing Vectors (zero vector check)')
    for i in range(numitems):
        if np.array_equal(vectors[i],zerovec):
            continue
        vec = vectors[i]
    #     print(vec)
        binval = struct.pack('f'*VECTOR_SIZE, *vec)
    #     print(binval)
        valStr = binascii.b2a_base64(binval, newline=False).decode('ascii')
        db[model][i] = valStr
        
    db.to_csv(TFIDF_DATA_DIR + TFIDF_OUTPUT_FILE, index=False, encoding='utf-8')    
    return(db) 


# In[6]:


# db = dataframe with defects and transformer binary vector
def find_tfidf_similar(model,db,MAX_SIMILAR):
    vectdict = {}
    cols =['Number']
    
    for i in range(MAX_SIMILAR):
        cols = cols + ['defectnum_' + str(i)] + ['defect_score_' + str(i)]
    df = pd.DataFrame()
    df[cols] = None
    df['Number'] = db['Number']
 
    print('Computing Vectors')
    
    zeroVec = list(np.zeros((VECTOR_SIZE,), dtype="float32"))
    for indx in db.index:
        binval = binascii.a2b_base64(db[model][indx])
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
            df['defectnum_' + str(i)][indx] = simNumber[i]
            df['defect_score_' + str(i)][indx] = simScore[i]
            
    df.to_csv(TFIDF_DATA_DIR + TFIDF_OUTPUT_DATA_FILE, index=False, encoding='utf-8')
    return(df)


# In[ ]:




