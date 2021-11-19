import argparse
import random
import os
import torch
import logging
import numpy as np
import sys
from transformers import BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from model.model import TrinityBert
import datetime
import multiprocessing
import pandas as pd
from sklearn import preprocessing
from util.util import seed_everything

sys.path.append("../")
sys.path.append("../..")
logger = logging.getLogger(__name__)


def get_train_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_folder", default="../../data/train", type=str,
                        help="The direcoty of the input training data files.")
    
    parser.add_argument("--data_file", default="../../data/train/train-0-20000.pkl", type=str,
                        help="The input training data file.")
    parser.add_argument("--vocab_file", default="../data/tags/commonTags_post2vec.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument(
        "--model_path", default=None, type=str,
        help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--logging_steps", type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_false",
                        help="Whether not to use CUDA when available")
    parser.add_argument("--valid_num", type=int, default=100,
                        help="number of instances used for evaluating the checkpoint performance")
    parser.add_argument("--valid_step", type=int, default=50,
                        help="obtain validation accuracy every given steps")
    parser.add_argument("--train_num", type=int, default=None,
                        help="number of instances used for training")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite the cached data")
    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
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
        "--max_steps", default=-1, type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--exe_name", type=str, help="name of this execution"
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--code_bert", default='microsoft/codebert-base',
                        choices=['microsoft/codebert-base', 'huggingface/CodeBERTa-small-v1',
                                 'codistai/codeBERT-small-v2'])
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    args = parser.parse_args()
    return args


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


def log_train_info(args, example_num, train_steps):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", example_num)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", train_steps)


def get_exe_name(args):
    exe_name = "{}_{}_{}"
    time = datetime.datetime.now().strftime("%m-%d %H-%M-%S")

    base_model = ""
    if args.model_path:
        base_model = os.path.basename(args.model_path)
    return exe_name.format(args.tbert_type, time, base_model)


def init_train_env(args, tbert_type):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
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
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set seed
    seed_everything(args.seed)
    
    # set one hot tokenizer for labels
    tab_vocab_path = args.vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    args.mlb = mlb
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    if tbert_type == 'trinity' or tbert_type == "T":
        model = TrinityBert(BertConfig(), args.code_bert)
    # elif tbert_type == 'siamese' or tbert_type == "I":
    #     model = TBertI(BertConfig(), args.code_bert)
    # elif tbert_type == 'siamese2' or tbert_type == "I2":
    #     model = TBertI2(BertConfig(), args.code_bert)
    # elif tbert_type == 'single' or tbert_type == "S":
    #     model = TBertS(BertConfig(), args.code_bert)
    else:
        raise Exception("TBERT type not found")
    args.tbert_type = tbert_type
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    return model



def train(args, train_examples, valid_examples, model, train_iter_method):
    """
    :param args:
    :param train_examples:
    :param valid_examples:
    :param model:
    :param train_iter_method: method use for training in each iteration
    :return:
    """
    
    # get the name of the execution
    if not args.exe_name:
        exp_name = get_exe_name(args)
    else:
        exp_name = args.exe_name
        
    # make output directory
    args.output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # get the train batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # the total number of batch per epoch
    train_numbers = len(train_examples) 
    epoch_batch_num = train_numbers  / args.train_batch_size
    
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (epoch_batch_num // args.gradient_accumulation_steps) + 1
    else:
        t_total = epoch_batch_num // args.gradient_accumulation_steps * args.num_train_epochs
        

    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    log_train_info(args, train_numbers, t_total)

    args.global_step = 0
    args.epochs_trained = 0
    args.steps_trained_in_current_epoch = 0
    if args.model_path and os.path.exists(args.model_path):
        ckpt = load_check_point(model, args.model_path, optimizer, scheduler)
        model = ckpt["model"]
        optimizer = ckpt['optimizer'] if ckpt['optimizer'] else optimizer
        scheduler = ckpt['scheduler'] if ckpt['scheduler'] else scheduler
        args = ckpt['args'] if ckpt['args'] else args
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch {}, global step {}".format(args.epochs_trained, args.global_step))
    else:
        logger.info("Start a new training")
    skip_n_steps_in_epoch = args.steps_trained_in_current_epoch  # in case we resume training
    model.zero_grad()
    train_iterator = trange(args.epochs_trained, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    step_bar = tqdm(initial=args.epochs_trained, total=t_total, desc="Steps")
    for _ in train_iterator:
        params = (
            args, model, train_examples, valid_examples, optimizer, scheduler, tb_writer, step_bar,
            skip_n_steps_in_epoch)

        # train_with_neg_sampling(*params)
        train_iter_method(*params)
        args.epochs_trained += 1
        skip_n_steps_in_epoch = 0
        args.steps_trained_in_current_epoch = 0

        if args.max_steps > 0 and args.global_step > args.max_steps:
            break

    model_output = os.path.join(args.output_dir, "final_model")
    save_check_point(model, model_output, args, optimizer, scheduler)
    step_bar.close()
    train_iterator.close()
    if args.local_rank in [-1, 0]:
        tb_writer.close()