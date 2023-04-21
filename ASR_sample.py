from utils import *
from models import *
import torch
import OpenAttack
import os
from contextlib import contextmanager
import pandas as pd
import random
import subprocess
import numpy as np
import pathlib

@contextmanager
def no_ssl_verify():
    import ssl
    from urllib import request

    try:
        request.urlopen.__kwdefaults__.update({'context': ssl.SSLContext()})
        yield
    finally:
        request.urlopen.__kwdefaults__.update({'context': None})

os.environ["WANDB_DISABLED"] = "true"
args = parse_train_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_lst = {'amazon_review_full':5,
               'amazon_review_polarity':2,'dbpedia':14,
               'yahoo_answers':10,'ag_news':4,
               'yelp_review_full':5,'yelp_review_polarity':2,
               'banking77__2':2, 'banking77__4':4, 'banking77__5':5, 
               'banking77__10':10, 'banking77__14':14,
               'tweet_eval_emoji_2':2, 'tweet_eval_emoji_4':4, 'tweet_eval_emoji_5':5, 
               'tweet_eval_emoji_10':10, 'tweet_eval_emoji_14':14,
              }

if __name__ == '__main__':
    file = open("generated_data/data_test.csv", "a")
    
    args.train_eval_sample = 'train'
    with open(f'generated_data/dataset_{args.chunk}.txt', 'r') as f:
        dataset = f.read()
    train_index = np.load(f'generated_data/train_index_{args.chunk}.npy')
    test_index = np.load(f'generated_data/test_index_{args.chunk}.npy')
    args.dataset, args.number_of_class = dataset, dataset_lst[dataset]
    model, tokenizer = get_model(args)
    train_data, test_data = load_dataset(args,train_index,test_index)

    # Train
    if args.model in ['roberta-base','bert-base']:
        train_data, test_data = preprocess_huggingface(args, tokenizer, train_data, test_data)
        model = model.to(args.device)
        if args.load_checkpoint == 'True':
            model.from_pretrained(pathlib.PurePath(args.load_dir))
        else:
            train_huggingface(args, model, train_data, train_data)
            model.save_pretrained(args.load_dir)


    # Evaluate
    del train_data
    del test_data
    torch.cuda.empty_cache()
    args.train_eval_sample = 'eval'
    test_index = np.load(f'generated_data/{args.dataset}_test_index.npy')
    attack_data = load_dataset(args,test_index=test_index)
    attack_data = preprocess_huggingface(args, tokenizer, test_data=attack_data)
    clsf = get_clsf(args, model, tokenizer)
    for attack_type in args.attack_type:
        print(f'\nAttacking with {attack_type}\n')
        with no_ssl_verify():
            if attack_type == 'TextFooler':
                attacker = OpenAttack.attackers.TextFoolerAttacker()
            elif attack_type == 'PWWS':
                attacker = OpenAttack.attackers.PWWSAttacker()
            elif attack_type == 'DeepWordBug':
                attacker = OpenAttack.attackers.DeepWordBugAttacker()
            elif attack_type == 'BERT':
                attacker = OpenAttack.attackers.BERTAttacker()

            attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics=[
                OpenAttack.metric.Fluency(),
                OpenAttack.metric.GrammaticalErrors(),
                OpenAttack.metric.SemanticSimilarity(),
                OpenAttack.metric.EditDistance(),
                OpenAttack.metric.ModificationRate()
            ] )

        summary = None
        retry_time = 0
        while summary is None and retry_time <= 2:
            try:
                summary = attack_eval.eval(attack_data, visualize=True, progress_bar=True)
                file.write(f'{summary["Attack Success Rate"]},')
            except:
                retry_time += 1
        if summary is None:
            file.write('Nan,')

        del attacker
        del attack_eval
        torch.cuda.empty_cache()