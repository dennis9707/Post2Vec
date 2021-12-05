import argparse
import logging
import os
import sys
import time
import logging
import os
import sys
from transformers import AutoTokenizer
from sklearn import preprocessing
import pandas as pd
import torch
sys.path.append("..")
from data_structure.question import Question, QuestionDataset
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from util.util import get_files_paths_from_directory

sys.path.append("..")
sys.path.append("../..")

import torch
from transformers import BertConfig
from model.model import TBertT
from train import get_tag_encoder
from train_trinity import load_data_to_dataset, get_dataloader
from util.eval_util import evaluate_batch

def avg(data):
    import numpy as np
    a = np.array(data)
    res = np.average(a, axis=0)
    return res

def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    train = pd.read_pickle(args.data_file)
    training_set = QuestionDataset(train, args.mlb, tokenizer)
    
    return training_set

def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../../data/test", type=str,
        help="The input test data dir.")
    parser.add_argument("--model_path", default="../../data/results/trinity_12-02 15-31-04_t_bert.pt/final_model-304/t_bert.pt", help="The model to evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--vocab_file", default="../../data/tags/commonTags_post2vec.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument("--output_dir", default="./results/evaluate/", help="directory to store the results")
    parser.add_argument("--code_bert", default="microsoft/codebert-base", help="the base bert")
    args = parser.parse_args()
    return args

def test(args, model, test_set):
    batch_size = 250
    test_data_loader = get_dataloader(
            test_set, batch_size)
    with torch.no_grad():
        model.eval()
        fin_outputs = []
        fin_targets = []
        for batch_idx, data in enumerate(test_data_loader, 0):
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
            logger.info(len(fin_outputs))
    [pre, rc, f1, cnt] = evaluate_batch(
        fin_outputs, fin_targets, [1, 2, 3, 4, 5]) 
    logger.info("F1 Score = {}".format(pre))
    logger.info("Recall Score  = {}".format(rc))
    logger.info("Precision Score  = {}".format(f1))
    logger.info("Count  = {}".format(cnt))
    return [pre, rc, f1, cnt]


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = get_eval_args()
    logger = logging.getLogger(__name__)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
     # get the encoder for tags
    mlb, num_class = get_tag_encoder(args.vocab_file)
    args.mlb = mlb
    args.num_class = num_class
    
    model = TBertT(BertConfig(), args.code_bert, num_class)
    model = torch.nn.DataParallel(model)
    model.to(device)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, )
        model.load_state_dict(torch.load(model_path),strict=False)
    logger.info("model loaded")
        # model.load_state_dict(torch.load(model_path))    
    fin_pre = []
    fin_rc = []
    fin_f1 = []
    fin_cnt = 0
    files = get_files_paths_from_directory(args.data_dir)

    start_time = time.time()
    
    for file_cnt in range(len(files)):
        # Load dataset and dataloader
        test_set = load_data_to_dataset(args.mlb, files[file_cnt])
        [pre, rc, f1, cnt] = test(args, model, test_set)
        fin_pre.append(pre)
        fin_rc.append(rc)
        fin_f1.append(f1)
        fin_cnt += cnt 
    
    avg_pre = avg(fin_pre)
    avg_rc = avg(fin_rc)
    avg_f1 = avg(fin_f1)
    logger.info("Final F1 Score = {}".format(avg_pre))
    logger.info("Final Recall Score  = {}".format(avg_rc))
    logger.info("Final Precision Score  = {}".format(avg_f1))
    logger.info("Final Count  = {}".format(fin_cnt))
    logger.info("Test finished")