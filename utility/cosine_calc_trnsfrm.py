import numpy as np
import struct
import binascii
import pandas as pd
import bisect
from numba import jit 

import time

INPUT_DIR = '../data/'
INPUT_DATA_FILE = 'transformers_vectors.csv'
OUTPUT_DATA_FILE = 'transformer_cosine_similarity.csv'

MAX_SIMILAR = 5
VECTOR_SIZE = 768
KEYS = ['Transformer_vector']


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
    
def find_similar(column):
    cols = ['score_0','score_1','score_2','score_3','score_4','similar_0','similar_1','similar_2','similar_3','similar_4',]
    
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

        df.iloc[indx] = [curNum] + simScore + simNumber
            
    df.to_csv(INPUT_DIR + OUTPUT_DATA_FILE, index=False, encoding='utf-8')
                
db = pd.read_csv(INPUT_DIR + INPUT_DATA_FILE)

for i in KEYS:
    find_similar(i)
