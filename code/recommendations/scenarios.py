# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:28:21 2022

@author: Alessandra
"""

import pandas as pd 
import random 
# Load data 
# gold = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)
pred = pd.read_csv("../../data/hlgd_predictions/best/preds_test_final.csv", index_col = 0)
all_urls = pred["url"].tolist()

# Get list of each gold story chain 
chain2 = pred.loc[pred["gold"] == 2]
chain4 = pred.loc[pred["gold"] == 4]
chain5 = pred.loc[pred["gold"] == 5]
chain6 = pred.loc[pred["gold"] == 6]
chain7 = pred.loc[pred["gold"] == 7]
chain8 = pred.loc[pred["gold"] == 8]
chain9 = pred.loc[pred["gold"] == 9]


# Get list of urls per chain 
url2 = chain2["url"].tolist()
url4 = chain4["url"].tolist()
url5 = chain5["url"].tolist()
url6 = chain6["url"].tolist()
url7 = chain7["url"].tolist()
url8 = chain8["url"].tolist()
url9 = chain9["url"].tolist()


# Get list of all predictions 
sBERT_AC = pred["SBERT_AC"].tolist()
sBERT_DB = pred["SBERT_DBScan"].tolist()
word_AC = pred["word_AC"].tolist()
word_DB = pred["word_DBScan"].tolist()
bow_AC = pred["BoW_AC"].tolist()
bow_DB = pred["BoW_DBScan"].tolist()
base = pred["baseline"].tolist()
gold = pred["gold"].tolist()


    
    

# Simulate users 
users = [i for i in range(1000)]

def df_to_list(chain): 
    """
    Transform DataFrame to a list containing each row as a list

    Parameters
    ----------
    chain : DataFrame
        containing all predictions per story chain 

    Returns
    -------
    row_list : list
        containing the 

    """ 
    
    row_list = []
    
    for rows in chain.itertuples():
        row = [rows.url, rows.SBERT_AC, rows.SBERT_DBScan, rows.word_AC, 
           rows.word_DBScan, rows.BoW_AC, rows.BoW_DBScan, rows.gold, rows.baseline]
        row_list.append(row)
    
    return row_list

row2 = df_to_list(chain2)
row4 = df_to_list(chain4)
row5 = df_to_list(chain5)
row6 = df_to_list(chain6)
row7 = df_to_list(chain7)
row8 = df_to_list(chain8)
row9 = df_to_list(chain9)



def get_random_sample(url): 
    """
    Get a random sample from a specified list of urls

    Parameters
    ----------
    url : list
        contains the urls for each story chain.

    Returns
    -------
    samples : list 
        extracts one sample for each user
        
    """
    samples = []
    for i in range(1000): 
        sample = random.sample(url, 1)
        samples.append(sample)
    return samples
    
sample2 = get_random_sample(url2)
sample4 = get_random_sample(url4)
sample5 = get_random_sample(url5)   
sample6 = get_random_sample(url6)
sample7 = get_random_sample(url7)
sample8 = get_random_sample(url8)
sample9 = get_random_sample(url9)


# ==========================================================
# ============= Scenario 1: Low Fragmentation ==============
# ==========================================================

# ===== 1a: 7 recommendations per user =====
low_frag_7 = pd.DataFrame()
low_frag_7["user"] = users


#for user in sample2: 
#  for url in user: 
#     for row in row2:
#         if item == row[0]:

# Add urls from each story chain to dataframe
urls_merged = []
for i, user in enumerate(users): 
    url_2 = sample2[i]
    url_4 = sample4[i]
    url_5 = sample5[i]
    url_6 = sample6[i]
    url_7 = sample7[i]
    url_8 = sample8[i]
    url_9 = sample9[i]
    urls_merged.append([url_2, url_4, url_5, url_6, url_7, url_8, url_9])
            

low_frag_7["urls"] = urls_merged

all_user_indeces = []
for user in urls_merged:
    user_index = []
    for link in user: 
        for url in link:
            index = all_urls.index(url)
            user_index.append(index)
    all_user_indeces.append(user_index)
    
low_frag_7["url_index"] = all_user_indeces

# Match user recommendation sets with the predictions per method

sBERT_AC = pred["SBERT_AC"].tolist()
sBERT_DB = pred["SBERT_DBScan"].tolist()
word_AC = pred["word_AC"].tolist()
word_DB = pred["word_DBScan"].tolist()
bow_AC = pred["BoW_AC"].tolist()
bow_DB = pred["BoW_DBScan"].tolist()
base = pred["baseline"].tolist()
gold = pred["gold"].tolist()



def get_recs_indeces(method): 

    all_recs_indeces = []
    for user in all_user_indeces: 
        recs_per_user = []
        for ind in user: 
            index = method[ind]
            recs_per_user.append(index)
        all_recs_indeces.append(recs_per_user)
        
    return all_recs_indeces

gold_recs = get_recs_indeces(gold)
sBERT_AC_recs = get_recs_indeces(sBERT_AC)
sBERT_DB_recs = get_recs_indeces(sBERT_DB)
word_AC_recs = get_recs_indeces(word_AC)
word_DB_recs = get_recs_indeces(word_DB)
bow_AC_recs = get_recs_indeces(bow_AC)
bow_DB_recs = get_recs_indeces(bow_DB)
base_recs = get_recs_indeces(base)

low_frag_7["gold"] = gold_recs
low_frag_7["baseline"] = base_recs
low_frag_7["SBERT_AC"] = sBERT_AC_recs
low_frag_7["SBERT_DB"] = sBERT_DB_recs
low_frag_7["word_AC"] = word_AC_recs
low_frag_7["word_DB"] = word_DB_recs
low_frag_7["bow_AC"] = bow_AC_recs
low_frag_7["bow_DB"] = bow_DB_recs

#low_frag_7.to_csv("../../data/recommendations/final_recs/scen1_low_frag_7.csv")

# ==========================================================
# ============= Scenario 3: High Fragmentation =============
# ==========================================================

high_frag_7 = pd.DataFrame()
high_frag_7["user"] = users

# Each user is linked to one story chain: all articles in their rec set are from this chain 
# 1 group of 142 users, 6 groups of 143

group2 = users[:142] # 142 users 
group4 = users[142:285] # 143 users 
group5 = users[285:428]
group6 = users[428:571]
group7 = users[571:714]
group8 = users[714:857]
group9 = users[857:1000]

# Give a random sample of 7 articles from the assigned story chain 

def get_high_sample(url, group): 
    """
    Get a random sample from a specified list of urls

    Parameters
    ----------
    url : list
        contains the urls for each story chain.

    Returns
    -------
    samples : list 
        extracts one sample for each user
        
    """
    samples = []
    for i in range(len(group)): 
        sample = random.sample(url, 7)
        samples.append(sample)
    return samples


group2samples = get_high_sample(url2, group2)
group4samples = get_high_sample(url4, group4)
group5samples = get_high_sample(url5, group5)
group6samples = get_high_sample(url6, group6)
group7samples = get_high_sample(url7, group7)
group8samples = get_high_sample(url8, group8)
group9samples = get_high_sample(url9, group9)

# Add recommended urls to dataframe 
merged_urls = group2samples + group4samples + group5samples + group6samples + group7samples + group8samples + group9samples
high_frag_7["urls"] = merged_urls

# Match articles with indeces
all_user_indeces = []
for user in merged_urls:
    user_index = []
    for link in user: 
        index = all_urls.index(link)
        user_index.append(index)
    all_user_indeces.append(user_index)
    
high_frag_7["url_index"] = all_user_indeces




sBERT_AC = pred["SBERT_AC"].tolist()
sBERT_DB = pred["SBERT_DBScan"].tolist()
word_AC = pred["word_AC"].tolist()
word_DB = pred["word_DBScan"].tolist()
bow_AC = pred["BoW_AC"].tolist()
bow_DB = pred["BoW_DBScan"].tolist()
base = pred["baseline"].tolist()
gold = pred["gold"].tolist()



def get_recs_indeces(method): 

    all_recs_indeces = []
    for user in all_user_indeces: 
        recs_per_user = []
        for ind in user: 
            index = method[ind]
            recs_per_user.append(index)
        all_recs_indeces.append(recs_per_user)
        
    return all_recs_indeces

gold_recs = get_recs_indeces(gold)
sBERT_AC_recs = get_recs_indeces(sBERT_AC)
sBERT_DB_recs = get_recs_indeces(sBERT_DB)
word_AC_recs = get_recs_indeces(word_AC)
word_DB_recs = get_recs_indeces(word_DB)
bow_AC_recs = get_recs_indeces(bow_AC)
bow_DB_recs = get_recs_indeces(bow_DB)
base_recs = get_recs_indeces(base)


high_frag_7["gold"] = gold_recs
high_frag_7["baseline"] = base_recs
high_frag_7["SBERT_AC"] = sBERT_AC_recs
high_frag_7["SBERT_DB"] = sBERT_DB_recs
high_frag_7["word_AC"] = word_AC_recs
high_frag_7["word_DB"] = word_DB_recs
high_frag_7["bow_AC"] = bow_AC_recs
high_frag_7["bow_DB"] = bow_DB_recs


high_frag_7.to_csv("../../data/recommendations/final_recs/scen2_high_frag_7.csv")
