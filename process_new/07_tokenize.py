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
    dataset = pd.read_csv(file)
    dataset = dataset.fillna('')
    file_name = file[25:]

    print("../new_data/tokenize_train/"+file_name, flush=True)
    q_list = list()
    cnt = 0
    for index, row in dataset.iterrows():
        qid = row['Id']
        title = row['Title']
        text = row['Body']
        code = row['Code']
        title_feature = gen_feature(title, title_max, tokenizer)
        text_feature = gen_feature(text, text_max, tokenizer)
        code_feature = gen_feature(code, code_max, tokenizer)
        tags = row['Tags']
        q_list.append(NewQuestion(qid, title_feature, text_feature, code_feature, "", tags))
        cnt += 1
        if cnt % 10000 == 0:
            print("process {}".format(cnt))
    import pickle
    with open("../new_data/tokenize_train/"+file_name, 'wb') as f:
        pickle.dump(q_list, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", default="../new_data/tokenize_csv")
    parser.add_argument("--out_dir", "-o", default="../new_data/tokenize_train")
    parser.add_argument("--model_type", default="Salesforce/codet5-base")
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
    input_path = args.input_dir
        
    file_paths = []
    for root, dirs, files in os.walk(input_path):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    file_paths.sort()
    print(file_paths)
    
    title_max = args.title_max
    text_max = args.text_max
    code_max = args.code_max
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    pbar = tqdm(total=len(files))
    update = lambda *args: pbar.update()
    pool = mp.Pool(80)
    # for file_path in file_paths:
    #     process_file_to_tensor(file_path, title_max, text_max, code_max, args, tokenizer)    
    
    for file_path in file_paths:
        pool.apply_async(process_file_to_tensor, args=(file_path, title_max, text_max, code_max, args, tokenizer), callback=update)
    pool.close()
    pool.join()
    logging.info("here")
    
    

if __name__ == "__main__":
    main()