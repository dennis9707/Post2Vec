import pandas as pd
import sys
sys.path.append("../bert")
sys.path.append("../..")
import logging
from data_structure.question import NewQuestion
import pandas as pd
from util.util import get_files_paths_from_directory
from tqdm import tqdm
import time
import multiprocessing as mp
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", local_files_only=True)
def gen_feature(tokens, max_length):

        feature = tokenizer(tokens, max_length=max_length,
                                 padding='max_length', return_attention_mask=True,
                                 return_token_type_ids=False, truncation=True,
                                 return_tensors='pt')
        res = {
            "input_ids": feature["input_ids"].flatten(),
            "attention_mask": feature["attention_mask"].flatten()}
        return res



dataframe = pd.read_csv("../data/tagdc_csv/Posts_50thoustand.csv")

import ast
for index, row in small.iterrows():
    tags = ast.literal_eval(row['Tags'])
    for t in tags:
        if t in small_tag_cnt_dict:
            small_tag_cnt_dict[t] += 1
        else:
            small_tag_cnt_dict[t] = 1
    if index % 10000 == 0:
        print(index)

import pickle
    with open(out_dir+file_name, 'wb') as f:
        pickle.dump(q_list, f)