import numpy as np
import pandas as pd
import gensim
import struct
import binascii
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

DATA_DIR = '../data/'
INPUT_DATA_FILE = 'angularjs_processed_withimagetext.csv'
OUTPUT_DATA_FILE = 'angularjs_w2v.csv'
MODEL_DIR = '../models/'
MODEL_FILES = ['glove-wiki-gigaword-300.bin','word2vec-google-news-300.bin','fasttext-wiki-news-subwords-300.bin']
W2V_COLS = ['W2V_glove','W2V_google','W2V_fasttext']
VECTOR_SIZE = 300

stop_words = set(stopwords.words('english'))

def sent2vec(model,s):

    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]

    featureVec = np.zeros((300,), dtype="float32")
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

def calcVectors():
  for i in range(len(MODEL_FILES)):
    df[W2V_COLS[i]] = None
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(MODEL_DIR + MODEL_FILES[i] ,binary=True)

    for indx in df.index:
      vec = sent2vec(word2vec,str(df['IssueText'][indx]))
      binval = struct.pack('f'*VECTOR_SIZE, *vec)
      valStr = binascii.b2a_base64(binval, newline=False).decode('ascii')
      df[W2V_COLS[i]][indx] = valStr
    print(f'Done with {MODEL_FILES[i]}')
    
    
df = pd.read_csv(DATA_DIR + INPUT_DATA_FILE)
columns = ['Title_new','Description_new','AttachmentText_new','Comments_new']
df[columns] = df[columns].fillna('')

df['IssueText'] = df[columns].agg(' '.join, axis=1)
calcVectors()
db= df[['Number'] + W2V_COLS]
db.to_csv(DATA_DIR + OUTPUT_DATA_FILE, index=False, encoding='utf-8')
    

