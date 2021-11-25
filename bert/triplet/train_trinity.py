import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
from util.util import save_check_point
from train import get_optimizer_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.util import get_files_paths_from_directory
from data_structure.question import Question, QuestionDataset
import torch
import pandas as pd
from sklearn import preprocessing
from transformers import AutoTokenizer
import os
import logging
from train import get_train_args, init_train_env
from torch.utils.tensorboard import SummaryWriter
import gc
from datetime import datetime
from apex.parallel import convert_syncbn_model, DistributedDataParallel as DDP
from util.eval_util import evaluate_batch
from model.loss import loss_fn
import numpy as np
from util.util import write_tensor_board


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
    args.global_step = 0
    for file_cnt in range(len(files)):
        # Load dataset and dataloader
        training_set = load_data_to_dataset(args.mlb, files[file_cnt])
        train_size = int(0.95 * len(training_set))
        valid_size = len(training_set) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            training_set, [train_size, valid_size])
        train_data_loader = get_dataloader(
            train_dataset, args.train_batch_size)
        valid_data_loader = get_dataloader(
            valid_dataset, args.train_batch_size)
        
        logger.info('############# FILE {}: Training Start   #############'.format(file_cnt))
        # Train!
        log_train_info(args)

        logger.info("Start a new training")
        # in case we resume training

        tr_loss = 0
        for epoch in range(args.num_train_epochs):
            logger.info('############# Epoch {}: Training Start   #############'.format(epoch))
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
            logger.info('############# Epoch {}: Training End     #############'.format(epoch))
            logger.info('############# Epoch {}: Validation Start   #############'.format(epoch))
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

                    fin_targets.extend(targets.cpu().detach().numpy().tolist())
                    fin_outputs.extend(torch.sigmoid(
                        outputs).cpu().detach().numpy().tolist())
            [pre, rc, f1, cnt] = evaluate_batch(
                fin_outputs, fin_targets, [1, 2, 3, 4, 5])
            logger.info("F1 Score = {}".format(pre))
            logger.info("Recall Score  = {}".format(rc))
            logger.info("Precision Score  = {}".format(f1))
            logger.info("Count  = {}".format(cnt))
            logger.info('############# Epoch {}: Validation End     #############'.format(epoch))

            logger.info("Training finished")
            # Save model checkpoint
        model_output = os.path.join(
            args.output_dir, "final_model-{}".format(file_cnt))
        save_check_point(model, model_output, args,
                            optimizer, scheduler)


if __name__ == "__main__":
    main()
