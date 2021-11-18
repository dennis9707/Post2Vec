# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     _2_remove_rare_tags_from_each_batch.py
   Input:          Questions.csv
   Output:
   Description:    Exclude quetions with all rare tags.
   Author :       
   date：          2021/11/06 4:16 PM
-------------------------------------------------
"""
from datetime import datetime
import sys
import pandas as pd
import ast
from util import load_tags
import argparse

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src/data_structure')


def filter_by_rare(target_tags, rare_tags):
    return list(set(target_tags) - set(rare_tags))


def filter_by_common(target_tags, common_tags):
    return list(set(target_tags).intersection(set(common_tags)))


def build_corpus(all_fpath, tags_vocab, corpus_fpath, isRare=False):
    rows = []
    print("Filtering corpus based on tags...")
    cnt = 0
    filter_cnt = 0
    df = pd.read_csv(all_fpath)
    for idx, row in df.iterrows():
        try:
            qid = row['Id']
            title = row['Title']
            desc_text = row['Body']
            desc_code = row['Code']
            creation_date = row['CreationDate']
            tags = ast.literal_eval(row['Tags'])
            if isRare == True:
                clean_tags = filter_by_rare(tags, tags_vocab)
            else:
                clean_tags = filter_by_common(tags, tags_vocab)

            if len(clean_tags) == 0:
                filter_cnt += 1
                continue
            try:
                row = {"qid": qid,
                       "title": title,
                       "desc_text": desc_text,
                       "desc_code": desc_code,
                       "creation_date": creation_date,
                       "clean_tags": clean_tags,
                       }
                rows.append(row)
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

    print("Starting Writing new CSV")
    cols = ["qid", "title", "desc_text",
            "desc_code", "creation_date", "clean_tags"]
    new_df = pd.DataFrame(rows, columns=cols)
    new_df.to_csv(corpus_fpath+".csv", index=False)
    new_df.to_pickle(corpus_fpath+".pkl")
    print("Write %s lines successfully." % cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input:
    source_corpus_fpath = "../data/questions/Questions54FilteredNA.csv"
    # rare_tags_fpath = "../data/tags/20211110/ts1000/_2_1_rareTags.csv"
    common_tags_fpath = "../data/tags/20211110/ts1000/_2_1_commonTags.csv"
    # Output:
    target_corpus_fpath = "../data/questions/Questions54TS1000"

    # rare_tags = load_tags(rare_tags_fpath)
    common_tags = load_tags(common_tags_fpath)
    build_corpus(source_corpus_fpath, common_tags, target_corpus_fpath)
