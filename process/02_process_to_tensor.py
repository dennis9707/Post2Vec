import sys
sys.path.append("../bert")
sys.path.append("../..")
import multiprocessing as mp
from typing import Counter
import argparse
import logging
import time
from tqdm import tqdm
from data_structure.question import Question, NewQuestion
import pandas as pd
from util.util import get_files_paths_from_directory
from tqdm import tqdm
import time
import multiprocessing as mp
from transformers import AutoTokenizer, PLBartTokenizer


def gen_feature(tokens, max_length, tokenizer):

        feature = tokenizer(tokens, max_length=max_length,
                                 padding='max_length', return_attention_mask=True,
                                 return_token_type_ids=False, truncation=True,
                                 return_tensors='pt')
        res = {
            "input_ids": feature["input_ids"].flatten(),
            "attention_mask": feature["attention_mask"].flatten()}
        return res

def process_file_to_tensor(file, title_max, text_max, code_max, args, tokenizer):
    out_dir = args.out_dir
    dataset = pd.read_pickle(file)
    file_name = file[24:]

    print(out_dir+file_name, flush=True)
    q_list = list()
    cnt = 0
    for question in dataset:
        qid = question.get_qid()
        title = question.get_title()
        title_feature = gen_feature(title, title_max, tokenizer)
        text = question.get_text()
        text_feature = gen_feature(text, text_max, tokenizer)
        code = question.get_code()
        code_feature = gen_feature(code, code_max, tokenizer)
        date = question.get_creation_date()
        tags = question.get_tag()
        q_list.append(NewQuestion(qid, title_feature, text_feature, code_feature, date, tags))
        cnt += 1
        if cnt % 10000 == 0:
            print("process {}".format(cnt))
    import pickle
    with open(out_dir+file_name, 'wb') as f:
        pickle.dump(q_list, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", default="../data/processed_train")
    parser.add_argument("--out_dir", "-o", default="../data/")
    parser.add_argument("--model_type", default='microsoft/codebert-base')
    parser.add_argument("--title_max", default=100)
    parser.add_argument("--text_max", default=512)
    parser.add_argument("--code_max", default=512)
    args = parser.parse_args()

    # If folder doesn't exist, then create it.
    import os
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print("created folder : ", args.out_dir)

    else:
        print(args.out_dir, "folder already exists.")
    
    title_max = args.title_max
    text_max = args.text_max
    code_max = args.code_max
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    files = get_files_paths_from_directory(args.input_dir)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    # process_file_to_tensor(files[0], title_max, text_max, code_max, args, tokenizer)
    pbar = tqdm(total=len(files))
    update = lambda *args: pbar.update()
    pool = mp.Pool(80)
    for file in files:
        pool.apply_async(process_file_to_tensor, args=(file, title_max, text_max, code_max, args, tokenizer), callback=update)
    pool.close()
    pool.join()
    logging.info("here")
    
    

if __name__ == "__main__":
    main()