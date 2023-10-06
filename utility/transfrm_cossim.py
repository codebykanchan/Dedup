from sentence_transformers import SentenceTransformer
#from numba import jit

import pandas as pd 
import numpy as np
import binascii
import struct

#import bisect
import time

DATA_DIR = '../data/'
INPUT_FILE = 'angularjs_processed_withimagetext.csv'
OUTPUT_FILE = 'transformers.csv'

MAX_SIMILAR = 5

df = pd.read_csv(DATA_DIR + INPUT_FILE)
df["FullText"] = df['Title_new'].fillna(' ').astype(str)+" "+ \
                 df['AttachmentText_new'].fillna(' ').astype(str)+" "+ \
                 df['Description_new'].fillna(' ').astype(str)+' '+ \
                 df['Comments_new'].fillna(' ').astype(str)

db = pd.DataFrame()
db['Number'] = df['Number']
db['Transformer_vector'] = None

model = SentenceTransformer('bert-base-nli-mean-tokens')

st = time.time()
sentence_embeddings = model.encode(df['FullText'].values.astype(str))
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
