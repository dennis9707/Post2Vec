import numpy as np
import pandas as pd
import csv
import time
import logging
import multiprocessing as mp
from tqdm import tqdm
import argparse
import os
from util import clean_html_tags, clean_html_tags2, separate_text_code
import ast

def process_csv(file_path,output_path, rare_tags):
    cnt = 0
    file_name = file_path[23:-4]
    logging.info(file_name)
    
    input_file = pd.read_csv(file_path)
    filter_cnt = 0
    cnt = 0
    q_list = []
    for idx, row in input_file.iterrows():
        try:
            qid = row['Id']
            title = row['Title']
            body = row['Body']
            code = row['Code']
            creation_date = row['CreationDate']
            tags = ast.literal_eval(row['Tags'])
            # remove rare tags
            clean_tags = list(set(tags) - set(rare_tags))
            if len(clean_tags) == 0:
                filter_cnt += 1
                continue
            try:
                q_list.append({"Id": qid,
                                "CreationDate": creation_date,
                                "Title": title,
                                "Body": body,
                                "Code": code,
                                "Tags": clean_tags,
                                })
                cnt += 1
            except Exception as e:
                print("Skip id=%s" % qid)
                print("Error msg: %s" % e)

            if cnt % 10000 == 0:
                print("Writing %d instances, filter %d instances..." % (cnt, filter_cnt))
        except Exception as e:
            print("Skip qid %s because %s" % (qid, e))
            filter_cnt += 1
    print("filter %d instances in total" % (filter_cnt))
    print("have %d instances in total" % (cnt))
    keys = q_list[0].keys()
    with open('../new_data/clean_data/'+file_name+'.csv', 'w', errors='surrogatepass') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(q_list)


def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML file to CSV')
    parser.add_argument('--input', '-p',  type=str, default="../new_data/post2023/", help='input directory path which stores splited csv')
    parser.add_argument('--output', '-o',  type=str, default="../new_data/clean_data/", help='output file path')
    args = parser.parse_args()
    input_path = args.input
    output_file = args.output
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    
    # get the file paths
    file_paths = []
    for root, dirs, files in os.walk(input_path):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    file_paths.sort()
    
    # print(file_paths)
    pbar = tqdm(total=len(file_paths))
    update = lambda *args: pbar.update()
    pool = mp.Pool(mp.cpu_count())
    logging.info("cpu count {}".format(mp.cpu_count()))
    start_time = time.time()

    rare_tags_fpath = "../new_data/tags/rare_tags.csv"
    rare_tags = []
    tag_df = pd.read_csv(rare_tags_fpath)
    for idx, row in tag_df.iterrows():
        rare_tags.append(row['tag'])
    print("# tags = %s" % len(rare_tags))
    # for target_file in file_paths:
    #     print(target_file)
    #     process_csv(target_file,output_file, rare_tags)
    #     break



    for target_file in file_paths:
        pool.apply_async(process_csv, args=(target_file,output_file, rare_tags), callback=update)
    pool.close()
    pool.join()
    
    logging.info("Time cost: {} s".format(str(time.time()-start_time)))

if __name__ == "__main__":
    main()