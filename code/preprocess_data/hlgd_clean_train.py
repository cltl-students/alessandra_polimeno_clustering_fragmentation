#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:37:45 2022

@author: alessandrapolimeno

Some URLS could not be extracted or parsed, or they contained noise that should
be filtered out. This script filters the training data of the HLGD dataset. 
"""

import pandas as pd

hlgd_train = pd.read_csv('../../../data/hlgd_texts_train.csv', index_col=0)


# ============= Remove recommended titles=============

def remove_text_allana(position):
    """
    Remove all titles that are appended to some articles starting with 
    'Allana'
    
    :param position: the index of the to-be-cleaned row 
    """
    # identify the to-be-cleaned text
    text = hlgd_train.loc[position, "text"]
    # identify the word from where the cleaning should start
    index = text.find('Allana') 
    # cut off the irrelevant text
    text = text[:index]
    # add the cleaned text to the dataframe
    hlgd_train.loc[position,"text"] = text
    print(position)
    print(text)
    print()
    print()

remove_list_a = [903, 842, 801, 767, 746, 737, 719, 712, 687, 559, 
               418, 341, 339, 304, 282, 249, 239, 106, 66, 65, 31, 16]

for number in remove_list_a: 
    remove_text_allana(number)


def remove_text_published(position):
    """
    Remove all titles that are appended to some articles starting with 
    'Published'
    
    :param position: the index of the to-be-cleaned row 
    """
    # identify the to-be-cleaned text
    text = hlgd_train.loc[position, "text"]
    # identify the word from where the cleaning should start
    index = text.find('Published')
    # cut off the irrelevant text
    text = text[:index]
    # add the cleaned text to the dataframe
    hlgd_train.loc[position,"text"] = text
    print(position)
    print(text)
    print()
    print()

remove_list_p = [842, 494]
for number in remove_list_p: 
    remove_text_published(number)
    
    
# ============= Drop unwanted articles =============

# This list contains articles consisting of CAPTCHA messages, and very long articles 
remove_list_c = [623, 579, 572, 569, 465, 374, 351, 346, 338, 307, 293,
                 266, 139, 125, 111, 110, 100, 40, 12, 508, 400, 581, 539]

hlgd_train = hlgd_train.drop(labels = remove_list_c, axis = 0)

# Save the cleaned dataset 
hlgd_train.to_csv('../../../data/hlgd_text_train2.csv', index = True)
