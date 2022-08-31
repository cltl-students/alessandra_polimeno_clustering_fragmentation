# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:25:46 2022

@author: Alessandra
"""

import pandas as pd

gold_dev = pd.read_csv('../../data/hlgd_predictions/gold_dev.csv', index_col=0)
gold = pd.read_csv('../../data/hlgd_predictions/gold_labels.csv', index_col=0)
dev_texts = pd.read_csv('../../data/hlgd_texts_dev.csv', index_col=0)
dev_texts = dev_texts.drop("gold_label", axis = 1)
texts = pd.read_csv('../../data/hlgd_texts.csv', index_col=0)
texts = texts.drop("gold_label", axis = 1)

# Merge gold labels with texts 
new_dev = dev_texts.merge(gold_dev, left_on = "url", right_on = "url")
new_train = texts.merge(gold, left_on = "url", right_on = "url")

# Extract all articles from chain 2  
chain3 = new_train.loc[new_train["gold_label"] == 3]
# Add them to the dev dataframe 
new_dev = new_dev.append(chain3, ignore_index = True)
# Remove them from train dataframe 
new_train = new_train[new_train["gold_label"] != 3]

# Save newly splitted data 
new_dev.to_csv("../../data/new_split/new_dev.csv")
new_train.to_csv("../../data/new_split/new_test.csv")


#new_dev.to_csv("../../data/hlgd_texts_dev.csv")
#new_train.to_csv("../../data/hlgd_texts.csv")