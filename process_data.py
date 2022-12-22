from utils import *
import os
args = parse_train_args()
args.train_eval = 'train'
dataset_dir = os.path.join(os.getcwd(),'datasets',args.dataset)
train_data, test_data = load_dataset(args)
train_data.to_csv(dataset_dir+f'/process_train_{args.dataset_size}.csv', index=False)
test_data.to_csv(dataset_dir+f'/process_test_{args.dataset_size}.csv', index=False)