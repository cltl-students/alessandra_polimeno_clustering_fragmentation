#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:11:15 2022

@author: alessandrapolimeno

This script extracts the HLGD dataset, scrapes the texts from the URLs, 
and performs some preprocessing on the extracted data. 
"""

from datasets import load_dataset
import pandas as pd
import trafilatura

def get_text_from_url(url):
    """
    Fetch the HTLM of a link, and extract the text
    
    :param url: string containing the URL to a webpage
    :return: a string containing the full text
    """
    
    processed_text = None
    downloaded = trafilatura.fetch_url(url)
    if type(downloaded) == str: 
        processed_text = trafilatura.extract(downloaded)

    return processed_text


# ============= Load dataset =============

data = load_dataset('hlgd', script_version="master")

# Get train, test and dev sets
hlgd_validation = data['validation']
hlgd_train = data['train']
hlgd_train = data['train']

# Read training data as dataframe
df_train = pd.DataFrame(hlgd_train)



# ============= URL extraction =============

# Remove all duplicate URLs, texts have to be extracted only once
urls = df_train['url_a'].tolist()
urlsb = df_train['url_b'].tolist()
for url in urlsb: 
    urls.append(url)
    
unique_urls = list(set(urls))


# Crawl the texts from the urls, save them in a dictionary
# key = url, value = text 

raw_texts = {}
for url in unique_urls: 
    text = get_text_from_url(url)
    raw_texts[url] = [text]


# Some processing: 
hlgd_texts = pd.DataFrame.from_dict(raw_texts, orient = 'index', columns = ['text'])
# Drop NAs
hlgd_texts = hlgd_texts.dropna()
# Write to csv 
hlgd_texts.to_csv('../../data/hlgd_texts_train.csv', index=True)

# Read again so that column names can be changed (This could not be done before
# because the df was constructed from a dict, and for some reason the url column
# could not be changed from 'index' to 'url'. However, this column has to be 
# retrieved in the following steps.)

hlgd_merge = pd.read_csv('../../data/hlgd_texts_train.csv', index_col=0)

# Give correct column names
hlgd_merge.columns = ['url', 'text']

# Save data with correct column names 
hlgd_merge.to_csv('../../data/hlgd_texts_train.csv', index=True)



#  ============= Filter irretrievable URLS  ============= 

retrievable_col1 = []
retrievable_col2 = []

url_merge = hlgd_merge['url'].tolist()
url_a = df_train['url_a'].tolist()
url_b = df_train['url_b'].tolist()

for link in url_a: 
    if link in url_merge:
        retrievable_col1.append(1)
    else:
        retrievable_col1.append(0)

for link in url_b: 
    if link in url_merge:
        retrievable_col2.append(1)
    else:
        retrievable_col2.append(0)


df_train['retrievable_col1'] = retrievable_col1
df_train['retrievable_col2'] = retrievable_col2

df_train = df_train[df_train['retrievable_col1'] == 1]
df_train = df_train[df_train['retrievable_col2'] == 1]
df_train = df_train.drop(['retrievable_col1', 'retrievable_col2'], axis = 1)


# Save filtered data set 
df_train.to_csv('../../data/hlgd_train_full.csv', index = True)



# ============= Get some stats =============

dates = df_train['date_a'].tolist()
dates2 = df_train['date_b'].tolist()
for date in dates2: 
    dates.append(date)

print("==================================")
print("Range of dates: ")    
print("First date: ", min(dates))
print("Last date: ", max(dates))
print("==================================")
print()


