# ClusteringFragmentation

This repository contains the following folders: 
## code 

#### clustering
- get_clusters.py contains the code for 3 text representation methods (Bag of Words, GloVe word embeddings, and Sentence-BERT sentence embeddings), as well as 2 clustering algorithms (agglomerative hierarchical clustering and DB-Scan) 
- hyperparam_tuning_AC.py and hyperparam_tuning_DB.py were used for hyperparameter tuning of both algorithms 
- get_baseline_clusters.py generates the baseline clusters (with TF-IDF representations and the Louvain Community Detection Algorithm) 

#### evaluation
- error_analysis.py contains the code that was used for the error analysis 
- get_best.py selects the best-performing systems and writes them to a new file
- get_frag.py generates news recommendation sets and calculates the Fragmentation Score for the experimental setups based on different scenarios 

#### preprocess_data
Contains all scripts that are used to explore, filter and split the data.
- hlgd_extraction.py contains the code with which the texts in the HeadLine Grouping Dataset can be scraped


## data
- Containing the evaluation and development set. Each row contains a URL, the corresponding text, and the gold label. The data consists of news articles that are scraped from the URLs provided by the HeadLine Grouping Dataset. 
## predictions 
- eval_scores.csv contains the values of the evaluation metrics for each experimental setup 
- preds_eval.csv contains the predicted label of each setup. 


