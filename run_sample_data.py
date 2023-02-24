from subprocess import call
from utils import *
import random
import numpy as np
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.INFO)

dataset_lst = [('amazon_review_full',5),
               ('amazon_review_polarity',2),('dbpedia',14),
               ('yahoo_answers',10),('ag_news',4),
               ('yelp_review_full',5),('yelp_review_polarity',2)]

args = parse_train_args()

if args.sample == 'sub_dataset':
    args.train_eval_sample = 'sample_train'
    logging.info('Sample sub-datasets')
    for i in tqdm(range(10)):
        args.dataset, args.number_of_class = random.choices(dataset_lst,weights = [1, 
                                                                                  5,5,
                                                                                  1,5,
                                                                                  1,5], k=1)[0]
        train_index, test_index = load_dataset(args)
        with open(f'generated_data/train_index_{i}.npy', 'wb') as f:
            np.save(f, np.array(train_index))
        with open(f'generated_data/test_index_{i}.npy', 'wb') as f:
            np.save(f, np.array(test_index))
        with open(f'generated_data/dataset_{i}.txt', 'w') as f:
            f.write(str(args.dataset))
    args.train_eval_sample = 'sample_attack'
    
    # logging.info('Sample attack samples')
    # for dataset, _ in tqdm(dataset_lst):
    #     args.dataset = dataset
    #     args.limit_test = 100
    #     attack_index = load_dataset(args)
    #     with open(f'generated_data/{dataset}_test_index.npy', 'wb') as f:
    #         np.save(f, np.array(attack_index))
elif args.sample == 'data_info':
    with open("generated_data/data.csv", "w") as file:
        file.write('Dataset,Average number of tokens,Number of unique tokens,Minimum number of tokens,Maximum number of tokens,Number of cluster,Number of classes,ASR_TextFooler,ASR_PWWS,ASR_BERT,ASR_DeepWordBug\n')
    for i in tqdm(range(70,90)):
        call(["python", "data_insight.py",'--chunk',f'{i}'])
        call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','TextFooler','PWWS'])
        call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','BERT','DeepWordBug'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','PWWS'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','BERT'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','DeepWordBug'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','TextFooler'])
        with open("generated_data/data.csv", "a") as file:
            file.write('\n')