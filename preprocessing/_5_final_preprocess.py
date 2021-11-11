# read train csv
# read train csv
import pandas as pd
import ast
import numpy as np
import multiprocessing as mp
from util import vocab_to_index_dict, indexing_label

train = pd.read_csv(
    '../data/csv/filtered_tags/Question1_filtered_more.csv', lineterminator='\n')

tab_vocab_path = "../data/csv/tags/ts5/_2_1_commonTags.csv"
tag_vocab = pd.read_csv(tab_vocab_path)
tag_vocab.head()
tag_list = tag_vocab["tag"].astype(str).tolist()
label_vocab = vocab_to_index_dict(tag_list, ifpad=False)

count = 0
train_result = []
print("start")
for value in train["clean_tags"]:
    labels = indexing_label(ast.literal_eval(value))
    train_result.append(labels)
    count += 1
    if count % 10000 == 0:
        print(count)

print(len(train_result))
print(train_result[1])
