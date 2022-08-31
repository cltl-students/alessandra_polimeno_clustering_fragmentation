# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:07:12 2022

@author: Alessandra
"""

import pandas as pd

preds = pd.read_csv("../../data/hlgd_predictions/evaluation_DB_test/eval_scores_test.csv", index_col = 0)
clus = pd.read_csv("../../data/hlgd_predictions/evaluation_DB_test/preds_test.csv", index_col = 0)
data = pd.read_csv('../../data/hlgd_texts.csv', index_col=0)
true = data["gold_label"].tolist()

best_sbert = clus["SBERT_AC_(1, 2)"].tolist()
best_word = clus["word_AC_(1, 1)"].tolist()
best_bow = clus["BoW_AC_(7, 1)"].tolist()
best_preds = pd.DataFrame()
best_preds["SBERT_DB"] = best_sbert
best_preds["word_DB"] = best_word
best_preds["BoW_DB"] = best_bow
best_preds["gold"] = true 

best_preds.to_csv("../../data/hlgd_predictions/evaluation_DB_test/best_test.csv")
