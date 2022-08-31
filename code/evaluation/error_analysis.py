# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:03:23 2022

@author: Alessandra
"""

import pandas as pd

full_data = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)

data = pd.read_csv("../../data/hlgd_predictions/best/preds_test_final.csv", index_col = 0)

gold = set(data["gold"].tolist())
sbert_ac = set(data["SBERT_AC"].tolist())
sbert_db = set(data["SBERT_DBScan"].tolist())
word_ac = set(data["word_AC"].tolist())
glove_db = set(data["word_DBScan"].tolist())

# GOLD 
c2 =  data.loc[data['gold'] == 2]
c4 =  data.loc[data['gold'] == 4]
c5 =  data.loc[data['gold'] == 5]
c6 =  data.loc[data['gold'] == 6]
c7 =  data.loc[data['gold'] == 7]
c8 =  data.loc[data['gold'] == 8]
c9 =  data.loc[data['gold'] == 9]

sbert_ac = c9["SBERT_AC"].tolist()
word_ac = c9["word_AC"].tolist()

# sbert: 4
# word: 1
wrong = []
for pred in sbert_ac: 
    if pred != 3: 
        wrong.append(pred)


mistakes = []
for tup in zip(sbert_ac, word_ac): 
    if tup[0] != 3 and tup[1] != 6: 
        mistakes.append(tup)
        

# SBERT_AHC 
s0 = data.loc[data["SBERT_AC"] == 0]
s1 = data.loc[data["SBERT_AC"] == 1]
s2 = data.loc[data["SBERT_AC"] == 2]
s3 = data.loc[data["SBERT_AC"] == 3]
s4 = data.loc[data["SBERT_AC"] == 4]
s5 = data.loc[data["SBERT_AC"] == 5]
s6 = data.loc[data["SBERT_AC"] == 6]
s7 = data.loc[data["SBERT_AC"] == 7]
s8 = data.loc[data["SBERT_AC"] == 8]

# WORD_AC 
word0 = data.loc[data["word_AC"] == 0]
word1 = data.loc[data["word_AC"] == 1]
word2 = data.loc[data["word_AC"] == 2]
word3 = data.loc[data["word_AC"] == 3]
word4 = data.loc[data["word_AC"] == 4]
word5 = data.loc[data["word_AC"] == 5]
word6 = data.loc[data["word_AC"] == 6]
word7 = data.loc[data["word_AC"] == 7]
word8 = data.loc[data["word_AC"] == 8]


# SBERT_DB 
db0 = data.loc[data["SBERT_DBScan"] == 0]
db1 = data.loc[data["SBERT_DBScan"] == 1]
db2 = data.loc[data["SBERT_DBScan"] == 2]
db3 = data.loc[data["SBERT_DBScan"] == 3]
db4 = data.loc[data["SBERT_DBScan"] == -1]

# DB_GloVe
g0 = data.loc[data["word_DBScan"] == 0]
g1 = data.loc[data["word_DBScan"] == 1] # 776 
g2 = data.loc[data["word_DBScan"] == 2] # 781 


inspect_texts = []
nums = [0, 27, 41, 52, 100, 111, 113, 118, 175, 181, 193, 197, 199, 275, 317, 
        499, 503, 575, 591, 633, 682, 700, 705, 722, 741, 765, 808, 824, 929, 968, 1002, 1008]
for num in nums: 
    df = full_data.iloc[num]
    text = df["text"]
    inspect_texts.append(text)

for text in inspect_texts:
    print(text)
    print()
    

t776 = full_data.iloc[776]
print(t776["text"])


t781 = full_data.iloc[781]
print(t781["text"])

index_8 = s8.index.tolist()
texts = []
for index in index_8: 
    text = full_data.iloc[[index]]
    texts.append(text)
