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

### In case you have to download the SpaCy word embeddings, run this: 
    
#from spacy.cli import download
# download("en_core_web_md") 



# =================== Load data ===================
data = pd.read_csv('data/hlgd_eval.csv', index_col=0)
#data = pd.read_csv('../../data/new_split/new_dev.csv', index_col=0)
urls = data['url'].tolist()
sentences = data['text'].tolist()




# =================== Get embeddings =================== 

def get_BoW(sentences): 
    CountVec = CountVectorizer(ngram_range=(1,1), stop_words='english')
    Count_data = CountVec.fit_transform(sentences)
    bow_vectors = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
    
    return bow_vectors


def get_representation(sentences, method = "word"):     
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
    

def get_clusters(embeddings, method, alg):
    
    
    if alg == "AC":

        if method == "BoW" :
            distance_threshold = 124
        if method == "word": 
            distance_threshold = 4
        if method == "SBERT": 
            distance_threshold = 5 
            
        clustering_model = AgglomerativeClustering(n_clusters = None, linkage = 'ward', distance_threshold = distance_threshold) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_
        
        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
        
            clustered_sentences[cluster_id].append(sentences[sentence_id])
        
    if alg == "DBScan": 
        
        if method == "SBERT": 
            ep = 1
            min_samples = 32
        if method == "BoW": 
            ep = 7
            min_samples = 1
        if method == "word": 
            ep = 1
            min_samples = 1
        
        clustering = DBSCAN(eps=ep, min_samples=min_samples).fit(embeddings)
        cluster_assignment = clustering.labels_
    
    return cluster_assignment



# ============ Represent the articles with the desired method ============
# Options: SBERT (sentence embeddings), word (word embeddings), BoW (Bag of Words)
methods = ["SBERT", "word", "BoW"]
methods = ["SBERT"]
algs = ["AC", "DBScan"]
algs = ["AC"]
pred_clusters = pd.DataFrame(urls, columns = ['url'])

#distance_thresholds = [4,5,6,7,8,9,124,134,144]


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
        
        for alg in algs:
       
            clusters = get_clusters(embeddings, method, alg = alg)
            #print(set(clusters))
            print(f"Save {method} clustering outcome...")
            pred_clusters[f'{method}_{alg}'] = clusters
            
            # Calculate V-measure 
            
            pred = clusters
            label = f"{method}_{alg}"
            hcv = homogeneity_completeness_v_measure(true, pred)
            if len(set(clusters)) > 1: 
                sil = silhouette_score(embeddings, clusters, metric="euclidean")
                db = davies_bouldin_score(embeddings, clusters)
                ch = calinski_harabasz_score(embeddings, clusters)
            else: 
                sil = 0
                db = 0
                ch = 0
                
            df = pd.DataFrame(columns = ["model", "homogeneity", "completeness", "v_measure", "sil", "db", "ch"])
            
            add = [label, hcv[0], hcv[1], hcv[2], sil, db, ch]
            
            labels.append(label)
            homs.append(hcv[0])
            comps.append(hcv[1])
            v_measures.append(hcv[2])
            sils.append(sil)
            dbs.append(db)
            chs.append(ch)


eval_scores = pd.DataFrame()
eval_scores["model"] = labels 
eval_scores["hom"] = homs
eval_scores["comp"] = comps
eval_scores["v_measure"] = v_measures
eval_scores["sil"] = sils 
eval_scores["db"] = dbs 
eval_scores["ch"] = chs   

print()
print("===================================================================")
print("All done!")
print("===================================================================")


pred_clusters["gold"] = true


chain2 = pred_clusters.loc[pred_clusters["gold"] == 2]
chain4 = pred_clusters.loc[pred_clusters["gold"] == 4]
chain5 = pred_clusters.loc[pred_clusters["gold"] == 5]
chain6 = pred_clusters.loc[pred_clusters["gold"] == 6]
chain7 = pred_clusters.loc[pred_clusters["gold"] == 7]
chain8 = pred_clusters.loc[pred_clusters["gold"] == 8]
chain9 = pred_clusters.loc[pred_clusters["gold"] == 9]

    
pred_clusters.to_csv('predictions/preds_eval.csv', index = True)
eval_scores.to_csv("predictions/eval_scores.csv", index = True)

