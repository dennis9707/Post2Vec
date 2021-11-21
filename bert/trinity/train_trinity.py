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

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    args = get_train_args()
    model = init_train_env(args, tbert_type='trinity')  
    train(args, model)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
