from utils import *
import os
import pandas as pd
import numpy as np
from helper import AutoTokenizer

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