# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:58:09 2022

@author: Alessandra
"""

import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import itertools
from itertools import product
import numpy as np

### In case you have to download the SpaCy word embeddings, run this: 
    
#from spacy.cli import download
# download("en_core_web_md") 



# =================== Load data ===================
# data = pd.read_csv('../../data/new_split/new_dev.csv', index_col=0)

data = pd.read_csv('../../data/CIT_DeskDrop/shared_articles.csv', index_col=0)

urls = data['url'].tolist()
sentences = data['text'].tolist()
content_ids = data["contentId"].tolist()


# =================== Get embeddings =================== 

def get_BoW(sentences): 
    CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
    Count_data = CountVec.fit_transform(sentences)
    bow_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
    
    return bow_vectors


def get_representation(sentences, method):     
    if method == "SBERT":
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        embeddings = model.encode(sentences)
        
    if method == "word": 
        embeddings = []
        nlp = spacy.load('en_core_web_md')
        for sent in sentences: 
           doc = nlp(sent)
           embeddings.append(doc.vector) 
    
    if method == "BoW": 
        embeddings = get_BoW(sentences)
    
    return embeddings
    

def get_clusters(embeddings, method, comb, alg):
    
    
    if alg == "AC":
        
        distance_threshold = comb[0]
        linkage = comb[1]
            
        clustering_model = AgglomerativeClustering(n_clusters = None, 
                                                   linkage = linkage, 
                                                   distance_threshold = distance_threshold) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_
        
        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
        
            clustered_sentences[cluster_id].append(sentences[sentence_id])
        
    if alg == "DBScan": 
        
        eps = comb[0]
        min_samples = comb[1]
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        cluster_assignment = clustering.labels_
    
    return cluster_assignment



# ============ Represent the articles with the desired method ============
# Options: SBERT (sentence embeddings), word (word embeddings), BoW (Bag of Words)
methods = ["SBERT", "word", "BoW"]
#methods = ["word"]

pred_clusters = pd.DataFrame(urls, columns = ['url'])


distance_threshold = [i for i in np.arange(1, 150, 1)]
linkages = ["ward","complete","average","single"]

# get all combinations of eps and samples
combinations = list(product(distance_threshold, linkages))

true = data["gold_label"].tolist()

labels = []
homs = []
comps = []
v_measures = []
sils = []
dbs = []
chs = []


for method in methods:
    print("===================================================================")
    print(f"Get article representation with method {method}...")
    embeddings = get_representation(sentences, method = method)
    print(f"Performing agglomerative hierarchical clustering for {method} representations...")
    for comb in combinations: 
        clusters = get_clusters(embeddings, method, comb, alg = "AC")
        #print(set(clusters))
        #print(f"Save {method} clustering outcome...")
        pred_clusters[f'{method}_AC_{comb}'] = clusters
        
        # Calculate V-measure 
        
        pred = clusters
        label = f"{method}_AC_{comb}"
        hcv = homogeneity_completeness_v_measure(true, pred)
        if len(set(clusters)) > 1: 
            sil = silhouette_score(embeddings, clusters, metric="euclidean")
            db = davies_bouldin_score(embeddings, clusters)
            ch = calinski_harabasz_score(embeddings, clusters)
        else: 
            sil = 0
            db = 0
            ch = 0
            
        labels.append(label)
        homs.append(hcv[0])
        comps.append(hcv[1])
        v_measures.append(hcv[2])
        sils.append(sil)
        dbs.append(db)
        chs.append(ch)
        
print()
print("===================================================================")
print("All done!")
print("===================================================================")

eval_scores = pd.DataFrame()
eval_scores["model"] = labels 
eval_scores["hom"] = homs
eval_scores["comp"] = comps
eval_scores["v_measure"] = v_measures
eval_scores["sil"] = sils 
eval_scores["db"] = dbs 
eval_scores["ch"] = chs   


# =================== TO DO =================
# variate linkages clustering for word embeddings

pred_clusters.to_csv('../../data/hlgd_predictions/evaluation_AC_dev/preds_dev_3_7.csv', index = True)
eval_scores.to_csv("../../data/hlgd_predictions/evaluation_AC_dev/eval_scores_dev_3_7.csv", index = True)
# =============================================================================
# 
# best_sbert = pred_clusters["SBERT_AC_(1.0, 96.5)"].tolist()
# best_word = pred_clusters["word_AC_(0.5, 41.5)"].tolist()
# best_bow = pred_clusters["BoW_AC_(14.5, 0.5)"].tolist()
# best_preds = pd.DataFrame()
# best_preds["SBERT_DB"] = best_sbert
# best_preds["word_DB"] = best_word
# best_preds["BoW_DB"] = best_bow
# best_preds["gold"] = true 
# 
# best_preds.to_csv("../../data/hlgd_predictions/best_test.csv")
# 
# 
# 
# =============================================================================

#eval_scores.to_csv("../../data/hlgd_predictions/full_eval_DBscan_dev.csv")


#pred = pred_clusters["word_pred"]
#hcv = homogeneity_completeness_v_measure(true, pred)
    
# print(set(clusters))

# ============ Save ============ 
#pred_clusters.to_csv('../../data/hlgd_predictions/predictions_raw.csv', index = True)


