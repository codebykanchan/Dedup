from sentence_transformers import SentenceTransformer

import pandas as pd 
import numpy as np
import binascii
import struct
from numba import jit 
import bisect


INPUT_DIR = './data/'
VECTOR_FILE = 'transformers.csv'
SIMILARITY_FILE = 'transformer_cosine_similarity.csv'
VECTOR_SIZE = 768
MAX_SIMILAR = 5
COLUMN = 'Transformer_vector'

model = SentenceTransformer('bert-base-nli-mean-tokens')
db_vec = pd.read_csv(INPUT_DIR + VECTOR_FILE, encoding='utf-8')
db_sim = pd.read_csv(INPUT_DIR + SIMILARITY_FILE, encoding='utf-8')


vector_dict = {}
zero_vector = list(np.zeros((VECTOR_SIZE,), dtype="float32"))
for indx in db_vec.index:
    binary_values = binascii.a2b_base64(db_vec[COLUMN][indx])
    val = np.array(struct.unpack('f'*VECTOR_SIZE, binary_values))
    if list(val) == zero_vector:
        continue
    vector_dict[db_vec['Number'][indx]] = val
        
dict_keys = list(vector_dict.keys())

   
@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    '''
    calculates cosine similarity score between vectors
    '''
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

def get_model_vector(item):
    '''
    calculates vector using model on given text
    '''
    text = item['Title_new'].fillna(' ').astype(str)+" "+ \
           item['AttachmentText_new'].fillna(' ').astype(str)+" "+ \
           item['Description_new'].fillna(' ').astype(str)+' '+ \
           item['Comments_new'].fillna(' ').astype(str)
           
    sentence_embeddings = model.encode(text)
    vec = sentence_embeddings[0]
    binary_values = struct.pack('f'*VECTOR_SIZE, *vec)
    value_str = binascii.b2a_base64(binary_values, newline=False).decode('ascii')
    
    return value_str
    

def get_similar(vec):
    '''
    get 5 most similar issues
    '''
    binary_values = binascii.a2b_base64(vec)
    current_vec = np.array(struct.unpack('f'*VECTOR_SIZE, binary_values))
            
    similarity_scores = [-2.0]*MAX_SIMILAR
    similar_issue_numbers = [0]*MAX_SIMILAR


    for key in dict_keys:
        currScore = cosine_similarity_numba(current_vec, vector_dict[key])
        
        if [currScore] > similarity_scores:
            idx = bisect.bisect(similarity_scores, currScore)
            similarity_scores.insert(idx, currScore)
            similar_issue_numbers.insert(idx, key)
            similarity_scores = similarity_scores[1:]
            similar_issue_numbers = similar_issue_numbers[1:]

    similar_issue_numbers.reverse()
    similarity_scores.reverse()
    
    retLst = {}
    for i in range(MAX_SIMILAR):
        retLst[str(similar_issue_numbers[i])] = similarity_scores[i]
        
    return retLst
    
    
def save_data(index, vec, scores):
    '''
    saves data to .csv
    '''
    if(index not in db_vec['Number'].values):
        db_vec.loc[len(db_vec.index)] =[index, vec]
        similar_values = [index]
        similarity_scores = []
        similar_issue_numbers = []
        for key in scores.keys():
            similarity_scores.append(scores[key])
            similar_issue_numbers.append(key)
        similar_values = similar_values + similarity_scores + similar_issue_numbers
        db_sim.loc[len(db_sim.index)] = similar_values
        
        db_vec.to_csv(INPUT_DIR + VECTOR_FILE, index=False, encoding='utf-8')
        db_sim.to_csv(INPUT_DIR + SIMILARITY_FILE, index=False, encoding='utf-8')
        