# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:30:11 2022

@author: Alessandra
"""

import pandas as pd

hlgd_dev = pd.read_csv('../../../data/hlgd_texts_dev_unfiltered.csv', index_col=0)



texts = hlgd_dev["text"].tolist()

# =============================================================================
# 
# for num, text in enumerate(texts[:75]): 
#      print(num, ": ")
#      print(text)
#      print()
#      print()
# 
# 
# =============================================================================

def remove_text_published(position):
    """
    Remove all titles that are appended to some articles starting with 
    'Published'
    
    :param position: the index of the to-be-cleaned row 
    """
    # identify the to-be-cleaned text
    text = hlgd_dev.loc[position, "text"]
    # identify the word from where the cleaning should start
    index = text.find('Published')
    # cut off the irrelevant text
    text = text[:index]
    # add the cleaned text to the dataframe
    hlgd_dev.loc[position,"text"] = text
    print(position)
    print(text)
    print()
    print()
    
remove_list_a = [195, 189]

for number in remove_list_a: 
    remove_text_published(number)
    


def remove_text_vc_fund(position):
    """
    Remove all titles that are appended to some articles starting with 
    'Allana'
    
    :param position: the index of the to-be-cleaned row 
    """
    # identify the to-be-cleaned text
    text = hlgd_dev.loc[position, "text"]
    # identify the word from where the cleaning should start
    index = text.find('VC fund') 
    # cut off the irrelevant text
    text = text[:index]
    # add the cleaned text to the dataframe
    hlgd_dev.loc[position,"text"] = text
    print(position)
    print(text)
    print()
    print()
    
remove_list_b = [324]

for number in remove_list_b: 
    remove_text_vc_fund(number)
    

# This list contains articles consisting of CAPTCHA messages, and very long articles 
remove_list_c = [ 319, 315, 290, 271, 247, 230, 227, 209, 201, 163, 156, 152, 
                 144, 139, 129, 117, 90, 73, 63, 39, 18, 17, 7, 252, 212, 145, 121, 107, 70]

hlgd_dev = hlgd_dev.drop(labels = remove_list_c, axis = 0)

# Save the cleaned dataset 
hlgd_dev.to_csv('../../../data/hlgd_texts_dev.csv', index = True)
