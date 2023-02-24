import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from top2vec import Top2Vec
from utils import *
from helper import AutoTokenizer

args = parse_train_args()
args.train_eval_sample = 'train'
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased',model_max_length=args.max_length)
documents = ['done' for i in range (50)]
model = Top2Vec(documents,min_count=5,embedding_model='universal-sentence-encoder')
file = open("generated_data/data.csv", "a")
i = args.chunk
dataset_lst = {'amazon_review_full':5,
               'amazon_review_polarity':2,'dbpedia':14,
               'yahoo_answers':10,'ag_news':4,
               'yelp_review_full':5,'yelp_review_polarity':2}

with open(f'generated_data/dataset_{i}.txt','r') as f:
    args.dataset = f.read()
test_index = np.load(f'generated_data/test_index_{i}.npy')
train_index = np.load(f'generated_data/train_index_{i}.npy')
train_data, test_data = load_dataset(args,train_index,test_index)
documents = train_data['text'].tolist()
tokens = []
snt_len = []
for text in documents:
    snt_tokens = tokenizer(text)['input_ids']
    tokens += snt_tokens
    snt_len.append(len(snt_tokens))
hdbscan_args = {'min_cluster_size': 5,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'}
cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(model._embed_documents(documents,32))
labels = cluster.labels_
file.write(f'{args.dataset},{sum(snt_len)/len(snt_len)},{len(list(set(tokens)))},{min(snt_len)},{max(snt_len)},{len(list(set(labels)))},{dataset_lst[args.dataset]},')