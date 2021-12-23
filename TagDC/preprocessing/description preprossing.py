# -*- coding: utf-8 -*-
import sys
sys.path.append("../../bert")
from util.util import get_files_paths_from_directory
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from data_structure.question import Question, TagDCQuestion
import argparse
import logging
import time
import os
from tqdm import tqdm
import multiprocessing as mp
from util.util import get_files_paths_from_directory


stopwords_list = stopwords.words()
porter_stemmer = PorterStemmer()         #使用nltk.stem.porter的PorterStemmer方法提取单词的主干

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

def process_post(file):
    out_dir = "../../data/tagdc/"
    file_name = file[16:]
    data = pd.read_pickle(file)
    length = len(data)
    q_list = []
    for i in range(length):
        qid = data[i].get_comp_by_name("qid")
        title = data[i].get_comp_by_name("title")
        text = data[i].get_comp_by_name("desc_text")
        date = data[i].get_comp_by_name("creation_date")
        tags = data[i].get_comp_by_name("tags")
        desp = title + text
        desp = desp.split()
        filtered_words = [word for word in desp if word not in stopwords_list]
        stemmer_words = [porter_stemmer.stem(word) for word in filtered_words]
        res = []
        for word in stemmer_words:
            panduan = is_number(word)
            if panduan == False:
                res.append(word)
        q_list.append(TagDCQuestion(qid, res, date, tags))
    # write to pickle
    import pickle
    with open(out_dir+file_name, 'wb') as f:
        pickle.dump(q_list, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", default="../../data/train")
    parser.add_argument("--out_dir", "-o", default="../../data/tagdc")
    args = parser.parse_args()

    # If folder doesn't exist, then create it.
    if not args.out_dir:
        os.makedirs(args.out_dir)
        print("created folder : ", args.out_dir)

    else:
        print(args.out_dir, "folder already exists.")
    
    files = get_files_paths_from_directory(args.input_dir)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    pbar = tqdm(total=len(files))
    update = lambda *args: pbar.update()

    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    for file in files:
        pool.apply_async(process_post, args=(file, ), callback=update)
    pool.close()
    pool.join()

    logging.info("Time cost: {} s".format(str(time.time()-start_time)))
    logging.info("Start to write files...")
    logging.info("Done")
        
if __name__ == '__main__':
    main()

     
       
 

