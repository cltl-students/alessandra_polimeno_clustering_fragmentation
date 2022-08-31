# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:02:12 2022

@author: Alessandra
"""

import pandas as pd 

# ============= Load train and test data =============
train = pd.read_csv('../../data/hlgd_texts_train.csv', index_col=0)
test = pd.read_csv('../../data/hlgd_texts_test.csv', index_col=0)

# ============= Merge =============
merge = train.append(test, ignore_index = True)

# ============= Additional cleaning =============

merge = merge.drop(index = [16, 166, 384, 388, 520])

# ============= Save ============= 
merge.to_csv('../../data/hlgd_texts.csv', index = True)



