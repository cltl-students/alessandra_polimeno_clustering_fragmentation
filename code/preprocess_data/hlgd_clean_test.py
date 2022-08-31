# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:10:29 2022

@author: Alessandra
"""


import pandas as pd

hlgd_test = pd.read_csv('../../../data/hlgd_texts_test_unfiltered.csv', index_col=0)

texts = hlgd_test["text"].tolist()

# =============================================================================
# for num, text in enumerate(texts): 
#     print(num, ": ")
#     print(text)
#     print()
#     print()
# =============================================================================


def remove_text_vc_fund(position):
    """
    Remove all titles that are appended to some articles starting with 
    'Allana'
    
    :param position: the index of the to-be-cleaned row 
    """
    # identify the to-be-cleaned text
    text = hlgd_test.loc[position, "text"]
    # identify the word from where the cleaning should start
    index = text.find('VC fund') 
    # cut off the irrelevant text
    text = text[:index]
    # add the cleaned text to the dataframe
    hlgd_test.loc[position,"text"] = text
    print(position)
    print(text)
    print()
    print()
    
remove_list_a = [248, 236, 206, 175, 163, 107, 105, 75, 74, 61, 9]

for number in remove_list_a: 
    remove_text_vc_fund(number)
    
    
# ============= Drop unwanted articles =============

# This list contains articles consisting of CAPTCHA messages, and very long articles 
remove_list_b = [256, 255, 254, 239, 232, 228, 227, 219, 207, 205, 203, 202,
                 200, 193, 187, 183, 171, 169, 149, 146, 140, 136, 83, 84, 
                 80, 78, 60, 48, 46, 45, 41, 40, 35, 12, 10]

hlgd_test = hlgd_test.drop(labels = remove_list_b, axis = 0)

# Save the cleaned dataset 
hlgd_test.to_csv('../../../data/hlgd_texts_test.csv', index = True)
