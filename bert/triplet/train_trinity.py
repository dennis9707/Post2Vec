import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
from util.util import write_tensor_board
import numpy as np
from model.loss import loss_fn
from util.eval_util import evaluate_batch
from apex.parallel import convert_syncbn_model, DistributedDataParallel as DDP
from datetime import datetime
import gc
from torch.utils.tensorboard import SummaryWriter
from train import get_train_args, init_train_env
import logging
import os
from transformers import AutoTokenizer
from sklearn import preprocessing
import pandas as pd
import torch
from data_structure.question import Question, QuestionDataset
from util.util import get_files_paths_from_directory, avg
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from train import get_optimizer_scheduler
from util.util import save_check_point
from torch.utils.data.distributed import DistributedSampler



logger = logging.getLogger(__name__)


def get_exe_name(args):
    exe_name = "{}_{}_{}"
    time = datetime.now().strftime("%m-%d %H-%M-%S")

    base_model = ""
    if args.model_path:
        base_model = os.path.basename(args.model_path)
    return exe_name.format(args.tbert_type, time, base_model)


def log_train_info(args):
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)


def load_data_to_dataset(mlb, file):
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", local_files_only=True)
    train = pd.read_pickle(file)
    training_set = QuestionDataset(train, mlb, tokenizer)
    return training_set


def get_dataloader(dataset, batch_size):
    # sampler = DistributedSampler(dataset)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             )
    return data_loader


def get_distribued_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             )
    return data_loader




def validate(model, args, valid_data_loader):
    fin_outputs = []
    fin_targets = []
    valid_loss = 0
    with torch.no_grad():
        model.eval()
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
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(
                outputs).cpu().detach().numpy().tolist())
            
    [pre, rc, f1, cnt] = evaluate_batch(
                fin_outputs, fin_targets, [1, 2, 3, 4, 5])
    valid_loss = valid_loss / len(valid_data_loader)
    logger.info("Final F1 Score = {}".format(pre))
    logger.info("Final Recall Score  = {}".format(rc))
    logger.info("Final Precision Score  = {}".format(f1))
    logger.info("Final Count  = {}".format(cnt))
    logger.info("Valid Loss  = {}".format(valid_loss))
    return valid_loss

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    args = get_train_args()
    model = init_train_env(args, tbert_type='trinity')

    # multiple train file
    files = get_files_paths_from_directory(args.data_folder)

    # total training examples 10279014
    train_numbers = 9765063
    epoch_batch_num = train_numbers / args.train_batch_size
    t_total = epoch_batch_num // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)
    # get the name of the execution
    exp_name = get_exe_name(args)

    # make output directory
    args.output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("n_gpu: {}".format(args.n_gpu))
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    log_train_info(args)
    logger.info(
                '############# Epoch {}: Training Start   #############'.format(epoch))

    args.global_step = 0
    valid_loss_min = 0
    for epoch in range(args.num_train_epochs):    
        for file_cnt in range(len(files)):
            # Load dataset and dataloader
            training_set = load_data_to_dataset(args.mlb, files[file_cnt])
            if (file_cnt + 1) % 5 == 0:
                train_size = int(0.90 * len(training_set))
                valid_size = len(training_set) - train_size
                train_dataset, valid_dataset = torch.utils.data.random_split(
                    training_set, [train_size, valid_size])
            else:
                train_dataset = training_set
            
            if args.local_rank == -1:
                train_data_loader = get_dataloader(
                    train_dataset, args.train_batch_size)
                if (file_cnt + 1) % 5 == 0:
                    valid_data_loader = get_dataloader(
                    valid_dataset, args.train_batch_size)
            else: 
                train_data_loader = get_distribued_dataloader(
                    train_dataset, args.train_batch_size)
                if (file_cnt + 1) % 5 == 0:
                    valid_data_loader = get_distribued_dataloader(
                    valid_dataset, args.train_batch_size)

            logger.info(
                '############# FILE {}: Training Start   #############'.format(file_cnt))
            
            tr_loss = 0
            valid_loss_min = np.Inf
            model.train()
            model.zero_grad()
            for step, data in enumerate(train_data_loader):
                title_ids = data['titile_ids'].to(
                    args.device, dtype=torch.long)
                title_mask = data['title_mask'].to(
                    args.device, dtype=torch.long)
                text_ids = data['text_ids'].to(args.device, dtype=torch.long)
                text_mask = data['text_mask'].to(args.device, dtype=torch.long)
                code_ids = data['code_ids'].to(args.device, dtype=torch.long)
                code_mask = data['code_mask'].to(args.device, dtype=torch.long)
                targets = data['labels'].to(args.device, dtype=torch.float)
                outputs = model(title_ids=title_ids,
                                title_attention_mask=title_mask,
                                text_ids=text_ids,
                                text_attention_mask=text_mask,
                                code_ids=code_ids,
                                code_attention_mask=code_mask)

                loss = loss_fn(outputs, targets)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    args.global_step += 1

                    if args.logging_steps > 0 and args.global_step % args.logging_steps == 0:
                        tb_data = {
                            'lr': scheduler.get_last_lr()[0],
                            'loss': tr_loss / args.logging_steps
                        }
                        logger.info("tb_data {}".format(tb_data))
                        logger.info(
                            'Epoch: {}, Batch: {}， Loss:  {}'.format(epoch, step, tr_loss / args.logging_steps))
                        tr_loss = 0.0
            logger.info(
                '############# FILE {}: Training End     #############'.format(file_cnt))
            
            # validation
            if (file_cnt + 1) % 5 == 0:
                logger.info(
                '############# FILE {}: Validation Start    #############'.format(file_cnt))
                valid_loss = validate(model, args, valid_data_loader)
                logger.info(
                '############# FILE {}: Validation End    #############'.format(file_cnt))
            

        # Save model checkpoint
        model_output = os.path.join(
            args.output_dir, "final_model-{}".format(file_cnt))
        save_check_point(model, model_output, args,
                         optimizer, scheduler)


if __name__ == "__main__":
    main()
