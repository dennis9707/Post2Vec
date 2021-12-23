from datetime import datetime
import sys
sys.path.append("../bert")
import pandas as pd
import ast
from util import load_tags
from data_structure.question import NewQuestion
import math
import time
import logging
import multiprocessing as mp
from tqdm import tqdm
def filter_by_rare(target_tags, rare_tags):
    return list(set(target_tags) - set(rare_tags))


def filter_by_common(target_tags, common_tags):
    return list(set(target_tags).intersection(set(common_tags)))


def build_corpus(all_fpath, tags_vocab):
    print("Filtering corpus based on tags...")
    cnt = 0
    filter_cnt = 0
    file_name = all_fpath[24:-4]
    print("../data/tagdc_filter/"+file_name+".pkl")
    df = pd.read_csv(all_fpath,encoding = "ISO-8859-1")
    df = df.fillna('')
    rows = []
    for idx, row in df.iterrows():
        try:
            qid = row['Id']
            title = row['Title']
            desc_text = row['Body']
            desc_code = row['Code']
            creation_date = row['CreationDate']
            tags = ast.literal_eval(row['Tags'])
            clean_tags = filter_by_common(tags, tags_vocab)
            if len(clean_tags) == 0:
                filter_cnt += 1
                continue
            try:
                rows.append(NewQuestion(qid, title, desc_text, desc_code, creation_date, clean_tags))
                cnt += 1
            except Exception as e:
                print("Skip id=%s" % qid)
                print("Error msg: %s" % e)

            if cnt % 10000 == 0:
                print("Writing %d instances, filter %d instances... \n %s" %
                      (cnt, filter_cnt, datetime.now().strftime("%H:%M:%S")))
        except Exception as e:
            print("Skip qid %s because %s" % (qid, e))
            filter_cnt += 1
    import pickle
    with open("../data/tagdc_filter/"+file_name+".pkl", 'wb') as f:
            pickle.dump(rows, f)

def main():
    source_corpus_dir = "../data/tagdc_csv_split"
    rare_tags_fpath = "../data/tagdc_csv/tagdc_rareTags.csv"
    common_tags_fpath = "../data/tagdc_csv/tagdc_commonTags.csv"
    import os
    file_paths = []
    for root, dirs, files in os.walk(source_corpus_dir):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    file_paths.sort()
    pbar = tqdm(total=len(file_paths))
    update = lambda *args: pbar.update()
    pool = mp.Pool(mp.cpu_count())
    print("cpu count {}".format(mp.cpu_count()))
    start_time = time.time()
    rare_tags = load_tags(rare_tags_fpath)
    common_tags = load_tags(common_tags_fpath)
    for target_file in file_paths:
        pool.apply_async(build_corpus, args=(target_file, common_tags), callback=update)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()