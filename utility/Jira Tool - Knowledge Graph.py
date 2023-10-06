# %%
candidate_sentences = pd.read_csv("angularjs_processed_withimagetext.csv")
candidate_sentences.shape

# %%
candidate_sentences = pd.DataFrame().assign(Id=candidate_sentences['Id'], CommentsN=candidate_sentences['Comments_new'])
print(candidate_sentences)

# %%
candidate_sentences

# %%
doc = nlp("thank pull request look contribution open source project look pull request need contributor license agreement cla information open cla check pull request")

for tok in doc:
    print(tok.text, "...", tok.dep_)

# %%
def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(str(sent)):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5  
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]

# %%
get_entities("make changes to your pull request")

# %%
from tqdm import tqdm
entity_pairs = []

for i in tqdm(candidate_sentences["CommentsN"]):
    entity_pairs.append(get_entities(i))

# %%
entity_pairs[15:40]

# %%
def get_relation(sent):

    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", [pattern]) 

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)

# %%
get_relation("Please check the items below ")

# %%
relations = [get_relation(i) for i in tqdm(candidate_sentences['CommentsN'].astype(str))]

# %%
pd.Series(relations).value_counts()[:50]

# %%
# extract subject build Knowledge Graph
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
kg_df

# %%
# create a directed-graph from a dataframe
G=nx.from_pandas_edgelist(kg_df, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='red', edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# %%
##take one relation at a time.. Let’s start with the relation “thank”:

G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="thank"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos, font_weight='normal')
plt.show()

# %%
#next visualize the graph for the “think” relation:

G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="think"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='red', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# %%
#Let’s see the knowledge graph of another important predicate, i.e., the “duplicate”
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="duplicate"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='lightgreen', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# %%
#Let’s see the knowledge graph of another important predicate, i.e., the “look good”
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="look good"], "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='red', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()

# %%
