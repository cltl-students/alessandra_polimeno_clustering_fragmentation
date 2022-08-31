# Detecting Fragmentation in News Story Chains

This repository is part of the Research Master thesis "Diversifying News Recommendation Systems by Detecting Fragmentation in News Story Chains" by Alessandra Polimeno. The project was supervised by prof. dr. Antske Fokkens, Myrthe Reuver and Sanne Vrijenhoek. 

## Background 
This thesis contributes to a line of research that aims to develop measures for diversity in the context of personalized news recommendation systems. The focus lies on the 
Fragmentation metric, which measures the overlap in news story chains that users are exposed to in their personalized news recommendations. News story chains consist
of articles that report on the same action that took place at a specific time. 

This code allows you to transform texts into three types of representations: 
* A simple Bag of Words
* Word embeddings (GloVe) 
* Sentence embeddings (Sentence-BERT) 

Two clustering methods are applied to the articles to obtain the groups of news story chains: 
* Agglomerative Hierarchical Clustering 
* DB-Scan 

## Repository Overvies
What follows is an overview of the scripts that were used for data scraping and cleaning, obtaining machine-readable text representations, applying clustering algorithms, and evaluation, as well as the dataset and results. 

### `data` 
The dataset consists of news articles that are annotated with the news story chain that they belong to. They are scraped from the URLs provided by the [HeadLine Grouping Dataset](https://huggingface.co/datasets/hlgd).
The data is split into a development and evaluation set. Each row contains a URL, the corresponding text, and the gold label. 

### `code`
#### `preprocess_data`
Contains all scripts that are used to explore, filter and split the data.
- `hlgd_extraction.py` contains the code with which the texts in the HeadLine Grouping Dataset can be scraped

#### `clustering`
- `get_clusters.py` contains the code for 3 text representation methods (Bag of Words, GloVe word embeddings, and Sentence-BERT sentence embeddings), as well as 2 clustering algorithms (agglomerative hierarchical clustering and DB-Scan) 
- `hyperparam_tuning_AC.py` and `hyperparam_tuning_DB.py` were used for hyperparameter tuning of both algorithms 
- `get_baseline_clusters.py` generates the baseline clusters (with TF-IDF representations and the Louvain Community Detection Algorithm) 

#### `evaluation`
- `error_analysis.py` contains the code that was used for the error analysis 
- `get_best.py` selects the best-performing systems and writes them to a new file
- `get_frag.py` generates news recommendation sets and calculates the Fragmentation Score for the experimental setups based on different scenarios 

### `predictions` 
The result of the evaluation can be found in this folder. 
- `eval_scores.csv` contains the values of the evaluation metrics for each experimental setup 
- `preds_eval.csv` contains the predicted label of each setup. 
 

## Acknowledgements 
The Fragmentation metric that is implemented is developed by Sanne Vrijenhoek. 
For further reading see: 
- S. Vrijenhoek, M. Kaya, N. Metoui, J. M ̈oller, D. Odijk, and N. Helberger. Recommenders with a mission: assessing diversity in news recommendations. In Proceedings of the 2021 Conference on Human Information Interaction and Retrieval, pages 173–183, 2021.
