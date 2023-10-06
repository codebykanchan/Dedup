import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from numba import jit
import bisect


import os
from multiprocessing import Pool

DATA_DIR = '../data/'
INPUT_FILE = 'angularjs_processed_withimagetext.csv'
OUTPUT_FILE = 'tfidf_cosinesim.csv'
MAX_SIMILAR = 5

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


df = pd.read_csv(DATA_DIR + INPUT_FILE)
df["FullText"] = df['Title_new'].fillna(' ').astype(str)+" "+ \
                 df['AttachmentText_new'].fillna(' ').astype(str)+" "+ \
                 df['Description_new'].fillna(' ').astype(str)+' '+ \
                 df['Comments_new'].fillna(' ').astype(str)
                 
cols = ['Score_1','Score_2','Score_3','Score_4','Score_5','Sim_1','Sim_2','Sim_3','Sim_4','Sim_5']
db = pd.DataFrame()
db['Number'] = df['Number']
db[cols] = None

vectors = None
vectdict = {}
dictkeys=[]
results = np.array([None]*11*db.shape[0]).reshape(db.shape[0],11)

def calculate_vectors():
    global vectors
    global dictkeys
    global vectdict
    global db
    global results

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df['FullText'].values.astype(str))

    vectors = matrix.asformat('array')
    numitems = vectors.shape[0]
    zerovec = np.array([0.0]*vectors.shape[1])

    print('sanitizing Vectors (zero vector check)')
    for i in range(numitems):
        results[i][0] = db['Number'][i]
        if np.array_equal(vectors[i],zerovec):
            continue
        vectdict[db['Number'][i]]= vectors[i]
        
def find_similar(indx):
    global vectors
    global dictkeys
    global vectdict
    global results

    curNum = results[indx][0]
    if curNum not in dictkeys:
        return
    curVec = vectdict[curNum]
    
    simScore = [-2.0]*MAX_SIMILAR
    simNumber = [0]*MAX_SIMILAR
    #if indx%100 == 0:
    print(f'working for index {indx}')
    
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

    simScore.reverse()
    simNumber.reverse()
    return simScore + simNumber

def main():
    global vectors
    global dictkeys
    global vectdict
    global results
    global db
    
    workers = os.cpu_count()
    print(f'Number of CPU {workers}')
    calculate_vectors()
    dictkeys = list(vectdict.keys())
    
    indxes = list(db.index)
    chunks = [indxes[x:x+workers*20] for x in range(0,len(indxes),workers*20)]
    
    for chunk in chunks:
        with Pool(workers) as p:
            R = p.map(find_similar,chunk)
            pos = 0
            for i in chunk:
                if R[pos] is not None:
                    db.iloc[i] =  [db['Number'][i]] + R[pos]
                pos += 1
            
    db.to_csv(DATA_DIR + OUTPUT_FILE, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
