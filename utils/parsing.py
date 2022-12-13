from argparse import ArgumentParser
import os

def parse_train_args():
    
    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./test_run', help='Folder in which to save model and logs')
    parser.add_argument('--limit_train', type=int, default=1000, help='Number of samples for training set')
    parser.add_argument('--limit_test', type=int, default=100, help='Number of samples for testing set')
    parser.add_argument('--dataset', type=str, default='imdb', 
                                               choices=['ag_news','amazon_review_full','amazon_review_polarity',
                                                        'dbpedia','imdb','sogou_news','yahoo_answers',
                                                        'yelp_review_full','yelp_review_polarity'])
    parser.add_argument('--model', type=str, default='char_cnn', 
                                   help='Model for training or evaluation',
                                   choices=['roberta-base','char_cnn','word_cnn',
                                            'bert-base','bilstm','lstm',
                                            'rnn','birnn'])
    parser.add_argument('--max_length', type=int, default=512, help='Folder in which to save model and logs')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size for training')    
    parser.add_argument('--epoches', type=int, default=100, help='batch_size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate')    
    
    # Configuration of character-level CNN
    parser.add_argument('--max_length_char_cnn', type=int, default=1024, help='Folder in which to save model and logs')
    parser.add_argument('--kernel_size', type=list, default=[7,3], help='Folder in which to save model and logs')
    parser.add_argument('--dropout_input', type=float, default=0.1)
    parser.add_argument('--max_pool', type=int, default=3, help='Folder in which to save model and logs')
    parser.add_argument(
        "--alphabet",
        type=str,
        default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}",
        help='alphabet for encoding characters'
    )
    parser.add_argument("--extra_characters", type=str, default="")
    
    # Configuration of Embedding
    parser.add_argument('--embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    
    # Configuration of word-level CNN
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--kernel_num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('--static', action='store_true', default=True, help='fix the embedding')

    # Configuration of LSTM & BiLSTM & RNN & BiRNN
    parser.add_argument('--stacked_layers', type=int, default=2, help='Number of stacked layers in LSTM & RNN')

    args = parser.parse_args()

    if args.model == 'char_cnn':
        args.number_of_characters = len(args.alphabet)+len(args.extra_characters)
    
    check_constraint(args)    
    add_load_dir(args)

    match args.dataset:
        case 'ag_news':
            args.number_of_class = 4
        case 'amazon_review_full':
            args.number_of_class = 5
        case 'amazon_review_polarity':
            args.number_of_class = 2
        case 'dbpedia':
            args.number_of_class = 14 
        case 'imdb':
            args.number_of_class = 2
        case 'sogou_news':
            args.number_of_class = 5
        case 'yahoo_answers':
            args.number_of_class = 10
        case 'yelp_review_full':
            args.number_of_class = 5
        case 'yelp_review_polarity':
            args.number_of_class = 2
        
    return args

def add_load_dir(args):
    max_length = args.max_length if args.model != 'char_cnn' else args.max_length_char_cnn
    args.load_dir = os.path.join(args.log_dir, args.model) + f'_{args.dataset}_{max_length}'

def check_constraint(args):
    if args.model == 'char_cnn':
        if args.max_length_char_cnn != 1024:
            raise Exception('Wrong max length for char_cnn, should be 1024')
    elif args.model in ['lstm','bilstm','rnn','birnn']:
        if args.model in ['lstm','bilstm'] and (args.max_length < 10 or args.max_length > 12):
            raise Exception('Wrong max length, should be in range [10,12]')
        if args.model in ['rnn','birnn'] and args.max_length != 10:
            raise Exception('Max length should be 10')
        elif args.dataset not in ['ag_news','amazon_review_full','amazon_review_polarity',
                                  'dbpedia','sogou_news','yahoo_answers']:
            raise Exception('Choose wrong dataset')
        elif args.epoches < 100:
            raise Exception('Epoches should be greater than 100')
        elif args.learning_rate > 5e-4:
            raise Exception('Learning rate should lower than 5e-4')
    else:
        if args.max_length != 512:
            raise Exception('Wrong max length, should be 512')