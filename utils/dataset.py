import pandas as pd

def load_dataset(args):
    if args.dataset == 'imdb':
        import datasets
        train_data, test_data = datasets.load_dataset(args.dataset, split =[f'train[:{args.limit_train}]', f'test[{args.limit_test}:]'])
        if args.train_eval == 'train':
            return pd.DataFrame({'text':train_data['text'], 
                                 'label':train_data['label']}),\
                   pd.DataFrame({'text':test_data['text'], 
                                 'label':test_data['label']})
        else:
            return pd.DataFrame({'text':test_data['text'], 
                                 'label':test_data['label']})
    elif args.dataset == 'ag_news':
        names = ['label','text','description'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/ag_news/train.csv', names=names, index_col=False)
        df_test = pd.read_csv('datasets/ag_news/test.csv', names=names, index_col=False)
    elif args.dataset == 'amazon_review_full':
        names = ['label','text','review_text'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/amazon_review_full/train.csv', names=names, index_col=False)
        df_test = pd.read_csv('datasets/amazon_review_full/test.csv', names=names, index_col=False)        
    elif args.dataset == 'amazon_review_polarity':
        names = ['label','text','review_text'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/amazon_review_polarity/train.csv', names=names, index_col=False)
        df_test = pd.read_csv('datasets/amazon_review_polarity/test.csv', names=names, index_col=False)
    elif args.dataset == 'dbpedia':
        names = ['label','text','content'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/dbpedia/train.csv', names=names, index_col=False)
        df_test = pd.read_csv('datasets/dbpedia/test.csv', names=names, index_col=False)
    elif args.dataset == 'sogou_news':
        names = ['label','text','content'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text']
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/sogou_news/train.csv', names=names, index_col=False)
        df_test = pd.read_csv('datasets/sogou_news/test.csv', names=names, index_col=False)
    elif args.dataset == 'yahoo_answers':
        names = ['label','text','content','best_answer'] if args.model in ['lstm','bilstm','rnn','birnn'] \
           else ['label','title','text','best_answer']
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/yahoo_answers/train.csv', names=names, index_col=False)
        df_test = pd.read_csv('datasets/yahoo_answers/test.csv', names=names, index_col=False)
    elif args.dataset == 'yelp_review_full':
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/yelp_review_full/train.csv', names=['label','text'], index_col=False)
        df_test = pd.read_csv('datasets/yelp_review_full/test.csv', names=['label','text'], index_col=False)
    elif args.dataset == 'yelp_review_polarity':
        if args.train_eval == 'train':
            df_train = pd.read_csv('datasets/yelp_review_polarity/train.csv', names=['label','text'], index_col=False)
        df_test = pd.read_csv('datasets/yelp_review_polarity/test.csv', names=['label','text'], index_col=False)
    
    if args.train_eval == 'train':
        return df_train[['label','text']][df_train.notnull().all(1)].sample(frac=1).iloc[:args.limit_train].reset_index(drop=True),\
               df_test[['label','text']][df_test.notnull().all(1)].sample(frac=1).iloc[:args.limit_test].reset_index(drop=True)
    else:
        return df_test[['label','text']][df_test.notnull().all(1)].sample(frac=1).iloc[:args.limit_test].reset_index(drop=True)