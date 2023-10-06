import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from numba import jit
import bisect

DATA_DIR = '../data/'
INPUT_FILE = 'angularjs_processed_withimagetext.csv'
OUTPUT_FILE = 'tfidf_cosinesim_s.csv'
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
                 
cols = ['Sim_1','Score_1','Sim_2','Score_2','Sim_3','Score_3','Sim_4','Score_4','Sim_5','Score_5']
db = pd.DataFrame()
db['Number'] = df['Number']
db[cols] = None

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df['FullText'].values.astype(str))

vectors = matrix.asformat('array')
numitems = vectors.shape[0]
zerovec = np.array([0]*vectors.shape[1])
vectdict = {}

print('sanitizing Vectors (zero vector check)')
for i in range(numitems):
    if np.array_equal(vectors[i],zerovec):
        continue
    vectdict[db['Number'][i]]= vectors[i]
    
dictkeys = list(vectdict.keys())

st = time.time()

for indx in db.index:
    curNum = db['Number'][indx]
    if curNum not in dictkeys:
        continue
    curVec = vectdict[curNum]
    
    simScore = [-2.0]*MAX_SIMILAR
    simNumber = [0]*MAX_SIMILAR
    #if indx%100 == 0:
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

    simScore.reverse()
    simNumber.reverse()
    for i in range(MAX_SIMILAR):
        db['Score_' + str(i+1)][indx] = simScore[i]
        db['Sim_' + str(i+1)][indx] = simNumber[i]

et = time.time()
exec_time = et - st
print('Total Execution time for cosine_similarity:', exec_time, 'seconds')
        
print('Exporting output data to: ', DATA_DIR + OUTPUT_FILE)
db.to_csv(DATA_DIR + OUTPUT_FILE, index=False, encoding='utf-8')
