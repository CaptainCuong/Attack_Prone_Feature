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
               ('yelp_review_full',5),('yelp_review_polarity',2),
               ('banking77__2',2), ('banking77__4',4), ('banking77__5',5), 
               ('banking77__10',10), ('banking77__14',14),
               ('tweet_eval_emoji_2',2), ('tweet_eval_emoji_4',4), ('tweet_eval_emoji_5',5), 
               ('tweet_eval_emoji_10',10), ('tweet_eval_emoji_14',14)
              ]

args = parse_train_args()

if args.sample == 'sub_dataset':
    args.train_eval_sample = 'sample_train'
    logging.info('Sample sub-datasets')
    for i in tqdm(range(250)):
        args.dataset, args.number_of_class = random.choices(dataset_lst,weights = [5, 
                                                                                  1,5,
                                                                                  1,5,
                                                                                  5,1,
                                                                                  4,5,5,
                                                                                  5,5,
                                                                                  4,5,5,
                                                                                  5,5
                                                                                  ], k=1)[0]
        train_index, test_index = load_dataset(args)
        with open(f'generated_data/train_index_{i}.npy', 'wb') as f:
            np.save(f, np.array(train_index))
        with open(f'generated_data/test_index_{i}.npy', 'wb') as f:
            np.save(f, np.array(test_index))
        with open(f'generated_data/dataset_{i}.txt', 'w') as f:
            f.write(str(args.dataset))
    
    logging.info('Sample attack samples')
    args.train_eval_sample = 'sample_attack'
    for dataset, _ in tqdm(dataset_lst):
        args.dataset = dataset
        args.limit_test = 100
        attack_index = load_dataset(args)
        with open(f'generated_data/{dataset}_test_index.npy', 'wb') as f:
            np.save(f, np.array(attack_index))

    logging.info('Sample error-test samples')
    args.train_eval_sample = 'sample_attack'
    args.limit_test = 100
    for dataset, _ in tqdm(dataset_lst):
        args.dataset = dataset
        test_index = load_dataset(args)
        with open(f'generated_data/{dataset}_test_error_index.npy', 'wb') as f:
            np.save(f, np.array(test_index))

elif args.sample == 'data_info':
    with open("generated_data/data_test.csv", "w") as file:
        file.write('Index,Dataset,Average number of tokens,Number of unique tokens,Minimum number of tokens,Maximum number of tokens,\
Mean distance,Fisher ratio,CalHara Index,DaBou Index,Number of cluster,Pearson Med,Number of labels,Kurtosis,Misclassification rate,\
Number of classes,ASR_TextFooler,ASR_PWWS,ASR_BERT,ASR_DeepWordBug\n')
    for i in tqdm(range(250)):
        # call(["python", "data_insight.py",'--chunk',f'{i}'])
        call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','TextFooler','PWWS'])
        call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','BERT','DeepWordBug'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','PWWS'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','BERT'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','DeepWordBug'])
        # call(["python", "ASR_sample.py",'--chunk',f'{i}','--attack_type','TextFooler'])
        with open("generated_data/data_test.csv", "a") as file:
            file.write('\n')