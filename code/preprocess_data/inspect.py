# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:54:17 2022

@author: Alessandra
"""

import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt

data = pd.read_csv("../../data/hlgd_old/hlgd_texts.csv", index_col = 0)
texts = data["text"].tolist()

lengths = []
for text in texts: 
    if len(text) != 618368:
        lengths.append(len(text))

mean_length = mean(lengths)
min_length = min(lengths)
max_length = max(lengths)
std_length = std(lengths)

df = pd.DataFrame(lengths, columns = ["length"])

plt.hist(df['length'], bins = np.arange(min(lengths), max(lengths)))

c=0
for length in lengths:
    if length < 1000: 
        c+=1
    
