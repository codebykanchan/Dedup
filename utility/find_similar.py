import numpy as np
import struct
import binascii
import pandas as pd
import bisect
from numba import jit 

import time

INPUT_DIR = '../data/'
INPUT_DATA_FILE = 'angularjs_w2v.csv'

KEYS = ['W2V_glove','W2V_google','W2V_fasttext']
MAX_SIMILAR = 5
VECTOR_SIZE = 300

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
    
def cossim(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

#@jit(nopython=True)
def find_similar(column):
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
            
    df.to_csv(INPUT_DIR + column + '.csv', index=False, encoding='utf-8')
                
db = pd.read_csv(INPUT_DIR + INPUT_DATA_FILE)

for i in KEYS:
    find_similar(i)
