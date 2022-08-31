# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:41:14 2022

@author: Alessandra
"""

import pandas as pd 
import nltk
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
# from scipy.spatial.distance import squareform, pdist
import networkx as nx
import community


#df = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)
df = pd.read_csv("../../data/hlgd_texts.csv", index_col = 0)
texts = df["text"].tolist()
ids = df.index.values.tolist()


#df = df.iloc[[0, 1, 2, 4, 299, 349]]
#texts = df["text"].tolist()
#ids = df.index.values.tolist()


en = spacy.load('en_core_web_md')


# Count word frequencies
wordfreq = {}
for text in texts:
    tokens = nltk.word_tokenize(text)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1


most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)


# Obtain idf values for each word
word_idf_values = {}
for token in most_freq:
    doc_containing_word = 0
    for document in texts:
        if token in nltk.word_tokenize(document):
            doc_containing_word += 1
    word_idf_values[token] = np.log(len(texts)/(1 + doc_containing_word))
    
# Obtain tf values
word_tf_values = {}
for token in most_freq:
    sent_tf_vector = []
    for document in texts:
        doc_freq = 0
        for word in nltk.word_tokenize(document):
            if token == word:
                  doc_freq += 1
        word_tf = doc_freq/len(nltk.word_tokenize(document))
        sent_tf_vector.append(word_tf)
    word_tf_values[token] = sent_tf_vector
    
# Combine them into tfidf
tfidf_values = []
for token in word_tf_values.keys():
    tfidf_sentences = []
    for tf_sentence in word_tf_values[token]:
        tf_idf_score = tf_sentence * word_idf_values[token]
        tfidf_sentences.append(tf_idf_score)
    tfidf_values.append(tfidf_sentences)

# Save arrays in dataframe
tf_idf_model = np.asarray(tfidf_values)
tf_idf_model = np.transpose(tf_idf_model)    
tf_idf_df = pd.DataFrame(tf_idf_model)    

cosine_sim = cosine_similarity(tf_idf_model)
cosine_df = pd.DataFrame(cosine_sim)


### Transform matrix into long form (to allow for pairwise comparisons)

# get values from incorrect indeces
cosine_df_ids = cosine_df.index.values.tolist()

# replace newly given indeces with orininal indeces that link to articles
cosine_df = cosine_df.rename(index = dict(zip(cosine_df_ids, ids)))
cosine_df = cosine_df.rename(columns = dict(zip(cosine_df_ids, ids)))

long_form = cosine_df.unstack()

# rename columns and turn into a dataframe
long_form.index.rename(['id_a', 'id_b'], inplace=True)
long_form = long_form.to_frame('cosine').reset_index()

# Filter cosines lower than threshold
long_form = long_form[(long_form['cosine'] > 0.5)]
long_form = long_form[long_form.id_a != long_form.id_b]


# Get partitions


def identify(df):
       # calculate cosine similarity between documents
       # ids = [article.id for article in documents if not article.text == '']
       # cosines = self.cos.calculate_all_distances(ids)
       # if cosines:
       #df = pd.DataFrame(cosines)
       df = df.drop_duplicates()
       # create graph
       G = nx.from_pandas_edgelist(df, "id_a", "id_b", edge_attr="cosine")
       # create partitions, or stories
       partition = community.best_partition(G)
       return partition
   
partitions = identify(long_form)

print(partitions)


part_df = pd.DataFrame([partitions])


# Save 
part_df.to_csv("../../data/CIT_DeskDrop/clustered/hlgd_clusters.csv")

#tf_idf_df.to_csv("../../data/hlgd_predictions/baseline_test/tf_idf_reps_test.csv")
#cosine_df.to_csv("../../data/hlgd_predictions/baseline_test/cosine_scores_test.csv")



