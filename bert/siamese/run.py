from sklearn import metrics, preprocessing
import pandas as pd
from data_structure.question import Question
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from model.loss import loss_fn
import gc
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from util.util import save_ckp, load_ckp
import argparse
import os
import random
from datetime import datetime

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def main():
    ############################ model arguments settings ############################
    parser = argparse.ArgumentParser(
        description='Multi-label Classifier based on Bert-based Models')

    # Required parameters
    parser.add_argument("--train_data_file", default="../data/questions/Train_Questions54TS1000.pkl", type=str,
                        help="The input training data file.")
    parser.add_argument("--valid_data_file", default="../data/questions/Valid_Questions54TS1000.pkl", type=str,
                        help="The input training data file.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="The testing data file")

    parser.add_argument("--vocab_file", default="../data/tags/20211110/ts1000/_2_1_commonTags.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--epoch", default=3, type=int,
                        help="The number of epoch")
    parser.add_argument("--train_batch_size", default=96, type=int,
                        help="Batch size for training.")
    parser.add_argument("--valid_batch_size", default=96, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('-dropout', type=float, default=0.1,
                        help='the probability for dropout [default: 0.1]')
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--metric_threshold', type=float, default=0.5,
                        help="threshold to metric calculation")

    args = parser.parse_args()

    # Build Tag Vocab
    seed_everything(args.seed)
    tab_vocab_path = args.vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    args.mlb = mlb
    input_train = args.train_data_file
    input_valid = args.valid_data_file
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    train(input_train, input_valid, mlb, args)


if __name__ == '__main__':
    main()
