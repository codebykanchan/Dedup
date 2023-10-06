#!/usr/bin/env python
# coding: utf-8

# In[58]:


# !pip install sentence-transformers


# In[59]:


from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import time
import struct
import binascii
import bisect
from numba import jit 


# In[60]:


DATA_DIR = '../data/model_perf/'
INPUT_FILE = 'angularjs_processed_withimagetext.csv'
OUTPUT_FILE = 'transformer_vec.csv'
OUTPUT_DATA_FILE ='transformer_cos_sim.csv'
MAX_SIMILAR = 5
VECTOR_SIZE = 768
model = 'Transformer_vector'


# In[61]:


def create_transformer_vec(df,DATA_DIR,OUTPUT_FILE):
    db = pd.DataFrame()
    db['Number'] = df['Number']
    db['Transformer_vector'] = None
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    st = time.time()
    sentence_embeddings = model.encode(df['IssueText'].values.astype(str))
    et = time.time()
    print(f'total time taken {et-st}')
    print(sentence_embeddings.shape)
    VECTOR_SIZE = sentence_embeddings.shape[1]

    for indx in db.index:
        vec = sentence_embeddings[indx]
        binval = struct.pack('f'*VECTOR_SIZE, *vec)
        valStr = binascii.b2a_base64(binval, newline=False).decode('ascii')
        db.iloc[indx] = [db['Number'][indx]] + [valStr]
    
    db.to_csv(DATA_DIR + OUTPUT_FILE, index=False, encoding='utf-8')
    return(db)


# In[62]:


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


# In[69]:


# db = dataframe with defects and transformer binary vector
def find_similar(model,db,MAX_SIMILAR,DATA_DIR,OUTPUT_DATA_FILE):
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
            
    df.to_csv(DATA_DIR + OUTPUT_DATA_FILE, index=False, encoding='utf-8')
    return(df)

