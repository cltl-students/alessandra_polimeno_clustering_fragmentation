# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:57:22 2022

@author: Alessandra
"""

import pandas as pd 
# Load data 
# gold = pd.read_csv("../../data/new_split/new_test.csv", index_col = 0)
pred = pd.read_csv("../../data/hlgd_predictions/best/preds_test_final.csv", index_col = 0)
baseline = pd.read_csv("../../data/hlgd_predictions/baseline_test/full_pred_test.csv", index_col = 0)
baseline = baseline.merge(pred, left_index = True, right_index = True)
baseline = baseline.drop(columns = ["SBERT_AC", "SBERT_DBScan", "word_AC", "word_DBScan", "BoW_AC", "BoW_DBScan"])

pred = pred.merge(baseline, left_on = "url", right_on = "url")


# Add baseline to df
base_index = set(baseline.index)
base_pred = baseline["baseline_prediction"].tolist()
pred_index = set(pred.index)
difference = set(pred_index.difference(base_index))
new_base_chains = []
for i in range(10,79):
    new_base_chains.append(i)
new_base_dict = dict(zip(difference, new_base_chains))
add_base_dict = dict(zip(base_index, base_pred))
new_base_dict.update(add_base_dict)

pred["baseline"] = pd.Series(new_base_dict)

pred.to_csv('../../data/hlgd_predictions/best/preds_test_final.csv', index = True)