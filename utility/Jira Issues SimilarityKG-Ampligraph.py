# %%
#apply to jira dataset
import pandas as pd
df =pd.read_csv("angularjs_processed_withimagetext.csv")

# %%
df.head()

# %%
# Creating knowledge graph triples from tabular dataset
db_triples = []

for _, row in df.iterrows():
    issue_id = "ID" + str(row["Id"])
    comments = row["Comments_new"]   
    #print(comments) #testing
    comments_triples = [(issue_id, "hasComments", comments)]# [(issue_id, "hasComments", a) for a in comments]
    db_triples.extend(comments_triples)


# %%
"""
# Split dataset
"""

# %%
df_new = pd.DataFrame(db_triples)

# %%
df_new.values

# %%
df_new.iloc[:,:-1].values

# %%
num_test = int(len(df_new) * (20 / 100))
num_test

# %%
N=int(len(df_new)-num_test)
N

# %%
df_new.isnull().sum()

# %%
X_test = df_new.tail(len(df_new)-N)
X_train = df_new.head(N-1)

# %%
print('Train set size: ', X_train.shape)
print('Test set size: ', X_test.shape)

# %%
# Training knowledge graph embedding with ComplEx model
from ampligraph.latent_features import ComplEx

model = ComplEx(batches_count=100,
                seed=0,
                epochs=200,
                k=150,
                eta=5,
                optimizer='adam',
                optimizer_params={'lr':1e-3},
                loss='multiclass_nll',
                regularizer='LP',
                regularizer_params={'p':3, 'lambda':1e-5},
                verbose=True)

# %%
#db_triples = np.array(db_triples)
#model.fit(db_triples)
model.fit(np.array(X_train))

# %%
# Finding duplicates ISSUES (entities)
import numpy as np
from ampligraph.discovery import find_duplicates
X_train=np.array(X_train)
entities = np.unique(X_train[:, 0])
dups, _ = find_duplicates(entities, model, mode='entity', tolerance=0.3)
print(list(dups)[:5])

# %%
print(df[df.Id.isin((66008836 , 81611791 , 69404042))][['Comments']])

# %%
print(df[df.Id.isin((60824801, 44373049,45179795,40426308,25577598))][['Comments']])

# %%
print(df[df.Id.isin((225407498,204056779))][['Comments']])

# %%
print(df[df.Id.isin((125181540, 113276817))][['Comments']])

# %%
print(df[df.Id.isin((194995608, 22380874))][['Comments']])

# %%
print(df[df.Id.isin((51949739, 50822084))][['Comments']])

# %%
"""
### Performance Evaluation
"""

# %%
import pandas as pd
from ampligraph.datasets import load_from_csv
#df =pd.read_csv("mykg.csv")
data = load_from_csv('.', 'JiraKG.csv', sep=',') 

# %%
import numpy as np
ent = np.unique(np.concatenate([data[:, 0].astype(str), data[:, 1].astype(str)]))

# %%
print("------ ENTITIES ------  ")
ent

# %%
rel = np.unique(data[:, 2].astype(str))

# %%
print("------ RELATIONSHIPS ------  ")
rel

# %%
data[:, [3, 2]] = data[:, [2, 3]]

# %%
data

# %%
dataNew = data[:, [1,2, 3]]

# %%
dataNew

# %%
#Split the data into train and test set from ‘data’ such that test set has #number of samples equal to ‘num_test_samples’ and there are no duplicate entries
from ampligraph.evaluation import train_test_split_no_unseen 

# %%
num_test_samples = int(len(dataNew) * (20 / 100))

# %%
X = {}
X['train'], X['test'] = train_test_split_no_unseen(dataNew.astype(str),test_size=num_test_samples, allow_duplication=True) 

# %%
print('Train set size: ', X['train'].shape)
print('Test set size: ', X['test'].shape) 

# %%
# Training knowledge graph embedding with ComplEx model
from ampligraph.latent_features import ComplEx

ce_model = ComplEx(batches_count=100,
                seed=0,
                epochs=200,
                k=150,
                eta=5,
                optimizer='adam',
                optimizer_params={'lr':1e-3},
                loss='multiclass_nll',
                regularizer='LP',
                regularizer_params={'p':3, 'lambda':1e-5},
                verbose=True)

# %%
ce_model.fit(X['train'].astype(str))

# %%
"""
### Evaluate the embedding model on test data
"""

# %%
from ampligraph.evaluation import evaluate_performance
test_rank = evaluate_performance(X['test'], model=ce_model,use_default_protocol=True, verbose=True) 

# %%
"""
### compute and print metrics:

"""

# %%
from ampligraph.evaluation import mr_score
mr_score(test_rank)

# %%

from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score    
mrr = mrr_score(test_rank)
hits_10 = hits_at_n_score(test_rank, n=10)
print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
hits_3 = hits_at_n_score(test_rank, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(test_rank, n=1)
print("Hits@1: %.2f" % (hits_1))
hits_5 = hits_at_n_score(test_rank, n=5)
print("Hits@5: %.2f" % (hits_5))

# %%
"""

"""

# %%
