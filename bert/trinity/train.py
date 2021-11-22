import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
import torch
from torch.optim import AdamW
from transformers import BertConfig, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
import gc
from sklearn import metrics, preprocessing
from datetime import datetime
import pandas as pd
from apex.parallel import convert_syncbn_model, DistributedDataParallel as DDP
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

logger = logging.getLogger(__name__)

def get_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="../../data/train", type=str,
                        help="The direcoty of the input training data files.")

    parser.add_argument("--data_file", default="../../data/train/train-0-20000.pkl", type=str,
                        help="The input training data file.")
    parser.add_argument("--vocab_file", default="../../data/tags/commonTags_post2vec.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument(
        "--model_path", default=None, type=str,
        help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--logging_steps", type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("--valid_num", type=int, default=100,
                        help="number of instances used for evaluating the checkpoint performance")
    parser.add_argument("--valid_step", type=int, default=50,
                        help="obtain validation accuracy every given steps")

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


def log_train_info(args):
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed) = %d",
        args.train_batch_size
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )


def get_exe_name(args):
    exe_name = "{}_{}_{}"
    time = datetime.now().strftime("%m-%d %H-%M-%S")

    base_model = ""
    if args.model_path:
        base_model = os.path.basename(args.model_path)
    return exe_name.format(args.tbert_type, time, base_model)

def get_tag_encoder(vocab_file):
    tab_vocab_path = vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    return mlb, len(mlb.classes_)


def init_train_env(args, tbert_type):
    # Setup CUDA, GPU & distributed training

    # no_cuda: whether or not to use cuda
    # local_rank = 0
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
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


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    
    # get the encoder for tags
    mlb, num_class = get_tag_encoder(args.vocab_file)
    args.mlb = mlb
    args.num_class = num_class
    
    # get the model
    if tbert_type == 'trinity':
        model = TBertT(BertConfig(), args.code_bert, args.num_class)
    else:
        raise Exception("TBERT type not found")
    
    args.tbert_type = tbert_type
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    return model


def train(args, train_data_loader, valid_data_loader, model, optimizer, scheduler):
    """
    :param args:
    :param model:
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
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="../runs/{}".format(exp_name))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # model = convert_syncbn_model(model)
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)
    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        # model = DDP(model)

    
    # Train!
    log_train_info(args)

    args.steps_trained_in_current_epoch = 0
    logger.info("Start a new training")
    # in case we resume training
    
    tr_loss = 0
    for epoch in range(args.num_train_epochs):
        if args.local_rank in [-1, 0]:
            print('############# Epoch {}: Training Start   #############'.format(epoch))
        model.train()
        for step, data in enumerate(train_data_loader):
            title_ids = data['titile_ids'].to(args.device, dtype=torch.long)
            title_mask = data['title_mask'].to(args.device, dtype=torch.long)
            text_ids = data['text_ids'].to(args.device, dtype=torch.long)
            text_mask = data['text_mask'].to(args.device, dtype=torch.long)
            code_ids = data['code_ids'].to(args.device, dtype=torch.long)
            code_mask = data['code_mask'].to(args.device, dtype=torch.long)
            targets = data['labels'].to(args.device, dtype=torch.float)
            model.zero_grad()
            outputs = model(title_ids=title_ids,
                            title_attention_mask=title_mask,
                            text_ids=text_ids,
                            text_attention_mask=text_mask,
                            code_ids=code_ids,
                            code_attention_mask=code_mask)

            loss = loss_fn(outputs, targets)
            if args.fp16:
                try:
                    from apex import amp
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            else:
                loss.backward()
            tr_loss += loss.item()

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            args.global_step += 1
            
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and step % args.logging_steps == 0:
                tb_data = {
                    'lr': scheduler.get_last_lr()[0],
                    'loss': tr_loss / args.logging_steps
                }
                write_tensor_board(tb_writer, tb_data, args.global_step)
                print(
                    f'Epoch: {epoch}, Batch: {step}， Loss:  {tr_loss / args.logging_steps}')
                current_time = datetime.now().strftime("%H:%M:%S")
                print("Current Time =", current_time)
                tr_loss = 0.0

            # Save model checkpoint
            if args.local_rank in [-1, 0] and args.save_steps > 0 and (step+1) % args.save_steps == 0:
                # step invoke checkpoint writing
                ckpt_output_dir = os.path.join(
                    args.output_dir, "checkpoint-{}-{}".format(epoch, step))
                save_check_point(model, ckpt_output_dir,
                                args, optimizer, scheduler)
                
        # evaluation
        if args.local_rank in [-1, 0]:
            print('############# Epoch {}: Training End     #############'.format(epoch))
            print(
                '############# Epoch {}: Validation Start   #############'.format(epoch))
            model.eval()
            fin_targets = []
            fin_outputs = []
            with torch.no_grad():
                for batch_idx, data in enumerate(valid_data_loader, 0):
                    title_ids = data['titile_ids'].to(
                        args.device, dtype=torch.long)
                    title_mask = data['title_mask'].to(
                        args.device, dtype=torch.long)
                    text_ids = data['text_ids'].to(
                        args.device, dtype=torch.long)
                    text_mask = data['text_mask'].to(
                        args.device, dtype=torch.long)
                    code_ids = data['code_ids'].to(
                        args.device, dtype=torch.long)
                    code_mask = data['code_mask'].to(
                        args.device, dtype=torch.long)
                    targets = data['labels'].to(
                        args.device, dtype=torch.float)

                    outputs = model(title_ids=title_ids,
                                    title_attention_mask=title_mask,
                                    text_ids=text_ids,
                                    text_attention_mask=text_mask,
                                    code_ids=code_ids,
                                    code_attention_mask=code_mask)
                    # target = targets.cpu().detach().numpy().tolist()
                    # output = torch.sigmoid(
                    #     outputs).cpu().detach().numpy().tolist()
                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(torch.sigmoid(
                        outputs).cpu().detach().numpy().tolist())
            [pre, rc, f1, cnt] = evaluate_batch(
                fin_outputs, fin_targets, [1, 2, 3, 4, 5])
            print(f"F1 Score = {pre}")
            print(f"Recall Score  = {rc}")
            print(f"Precision Score  = {f1}")
            print(f"Count  = {cnt}")
            print(
                '############# Epoch {}: Validation End     #############'.format(epoch))
    return model, optimizer, scheduler
