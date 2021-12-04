import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
import torch
from torch.optim import AdamW
from transformers import BertConfig, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
import gc
from sklearn import preprocessing
from datetime import datetime
import pandas as pd
from util.eval_util import evaluate_batch
from util.util import seed_everything, save_check_point
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from model.loss import loss_fn
from model.model import TBertT
import numpy as np
import logging

import os
import random
import argparse
from data_structure.question import Question, QuestionDataset
import pandas as pd
from util.util import write_tensor_board
from util.data_util import get_tag_encoder

logger = logging.getLogger(__name__)

def get_exe_name(args):
    exe_name = "{}_{}_{}"
    time = datetime.now().strftime("%m-%d %H-%M-%S")

    base_model = ""
    return exe_name.format(args.tbert_type, time, base_model)

def get_optimizer_scheduler(args, model, train_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=train_steps
    )
    return optimizer, scheduler

def init_train_env(args, tbert_type):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "local rank %s, device: %s, n_gpu: %s",
        args.local_rank,
        device,
        args.n_gpu,
    )
    
    # get the encoder for tags
    mlb, num_class = get_tag_encoder(args.vocab_file)
    args.mlb = mlb
    args.num_class = num_class
    
    # Load pretrained model and tokenizer
    # if args.local_rank not in [-1, 0]:
    #     # Make sure only the first process in distributed training will download model & vocab
    #     torch.distributed.barrier()
    if tbert_type == 'trinity':
        model = TBertT(BertConfig(), args.code_bert, args.num_class)
    else:
        raise Exception("TBERT type not found")
    args.tbert_type = tbert_type
    # if args.local_rank == 0:
    #     # Make sure only the first process in distributed training will download model & vocab
    #     torch.distributed.barrier()
        
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    return model


def get_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="../../data/tensor_data", type=str,
                        help="The direcoty of the input training data files.")

    parser.add_argument("--data_file", default="../../data/train/train-0-20000.pkl", type=str,
                        help="The input training data file.")
    parser.add_argument("--vocab_file", default="../../data/tags/commonTags.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument(
        "--model_path", default="../../data/results/trinity_11-22 14-23-51_/final_model-156/t_bert.pt", type=str,
        help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--logging_steps", type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--output_dir", default="../results", type=str,
        help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--learning_rate", default=1e-6,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--code_bert", default='microsoft/codebert-base',
                        choices=['microsoft/codebert-base', 'huggingface/CodeBERTa-small-v1',
                                 'codistai/codeBERT-small-v2'])
    args = parser.parse_args()
    return args