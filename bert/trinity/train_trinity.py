from train import get_train_args, init_train_env, train
import logging
import os
import sys
from transformers import AutoTokenizer
from sklearn import preprocessing
import pandas as pd
import torch
sys.path.append("..")
from data_structure.question import Question, QuestionDataset
logger = logging.getLogger(__name__)

def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    train = pd.read_pickle(args.data_file)
    training_set = QuestionDataset(train, args.mlb, tokenizer)
    
    return training_set


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    args = get_train_args()
    model = init_train_env(args, tbert_type='trinity')
    training_set = load_data(args)
    
    train_size = int(0.95 * len(training_set))
    test_size = len(training_set) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(training_set, [train_size, test_size])    
    train(args, train_dataset, test_dataset, model)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
