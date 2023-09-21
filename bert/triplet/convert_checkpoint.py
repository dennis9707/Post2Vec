import argparse
import logging
import os
import sys
import time
import logging
import os
import sys
sys.path.append("..")
sys.path.append("../..")
import torch
from transformers import BertConfig, AutoConfig
from util.util import get_files_paths_from_directory
from model.model import TBertT,TBertSI, TBertTNoCode
from util.data_util import get_tag_encoder, get_fixed_tag_encoder, load_data_to_dataset, get_dataloader, load_tenor_data_to_dataset, load_data_to_dataset_for_test
from torch.utils.data import DataLoader
import numpy as np
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../../data/test", type=str,
        help="The input test data dir.")
    
    parser.add_argument("--model_path", default="../../data/results/triplet_12-30 06-12-15_/epoch-0-file-509/t_bert.pt", help="The model to evaluate")
    # parser.add_argument("--model_path", default="../../data/results/triplet_12-07 15-29-36_/final_model-199/t_bert.pt", help="The model to evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--vocab_file", default="../../data/tags/commonTags_post2vec.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument("--verbus", action="store_true", help="show more logs")
    parser.add_argument("--mlb_latest", action="store_true", help="use the latest mlb")
    parser.add_argument("--test_batch_size", default=500, type=int,help="batch size used for testing")
    parser.add_argument("--output_dir", default="./logs", help="directory to store the results")
    parser.add_argument("--code_bert", default='microsoft/codebert-base',
                        choices=['microsoft/codebert-base', 'huggingface/CodeBERTa-small-v1',
                                 'codistai/codeBERT-small-v2', 'albert-base-v2','jeniya/BERTOverflow', 'roberta-base',
                                 'bert-base-uncased'])
    parser.add_argument("--model_type", default="triplet", choices=["triplet","siamese"])
    args = parser.parse_args()
    return args
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_eval_args()
    logging.info("Start Testing") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    # get the encoder for tags
    if args.mlb_latest == True:
        logger.info("use new mlb tagger")
        mlb, num_class = get_fixed_tag_encoder(args.vocab_file)
    else:
        mlb, num_class = get_tag_encoder(args.vocab_file)
        
    print(num_class)
    args.mlb = mlb
    args.num_class = num_class
    # config = AutoConfig.from_pretrained(args.code_bert)
    if args.model_type == "triplet":
        model = TBertT(BertConfig(), args.code_bert, num_class)
    elif args.model_type == "siamese":
        model = TBertSI(BertConfig(), args.code_bert, num_class)
    model = torch.nn.DataParallel(model)
    model.to(device)
    
    if args.code_bert == "microsoft/codebert-base":
        args.model_path = "./epoch1_t_bert.pt"
        args.name = "codebert"
    elif  args.code_bert == "roberta-base":
        args.model_path = "../../data/results/triplet_01-02-02-57-44_/epoch-0-file-499/t_bert.pt"
        args.name = "roberta"
    elif  args.code_bert == "jeniya/BERTOverflow":
        args.model_path = "../../data/results/triplet_01-02-02-54-11_/epoch-0-file-500/t_bert.pt"
        args.name = "bertoverflow"

    elif  args.code_bert == "albert-base-v2":
        args.model_path = "../../data/results/albert-base-v2_01-02-06-19-49_/epoch-0-file-499/t_bert.pt"
        args.name = "albert"

    elif  args.code_bert == "bert-base-uncased":
        args.model_path = "../../data/results/bert-base-uncased_01-05-15-56-05_/epoch-0-file-499/t_bert.pt"
        args.name = "bert"

    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, )
        model.load_state_dict(torch.load(model_path)) 
    logger.info("model loaded")
    
    ckpt_dir = "../../checkpoint"
    logger.info("Saving checkpoint to %s", ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Check if model was parallelized with DataParallel or DistributedDataParallel
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        torch.save(model.module.state_dict(), os.path.join(ckpt_dir, args.name))
    else:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, args.name))
if __name__ == "__main__":
    main()