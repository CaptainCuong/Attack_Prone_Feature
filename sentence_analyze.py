from utils import *
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

args = parse_train_args()
if __name__ == '__main__':
    dataset_dir = os.path.join(os.getcwd(),'datasets',args.dataset)
    test_data = pd.read_csv(dataset_dir+f'/process_train_{args.dataset_size}.csv', index_col=False)
    documents = test_data['text'].tolist()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased',model_max_length=args.max_length)
    len_lst = []
    for text in documents:
        len_lst.append(len(tokenizer(text)['input_ids']))
    print(sum(len_lst)/len(len_lst))
    print(min(len_lst))
    print(max(len_lst))
    plt.hist(len_lst,bins=100)
    plt.suptitle(f'Distribution of the numbers of tokens in sentences in dataset {args.dataset}')
    plt.ylabel('Number of sentences')
    plt.xlabel('Number of tokens in each sentence')
    plt.savefig(dataset_dir+f'/image_train_token_{args.dataset_size}.png')
    plt.show() 