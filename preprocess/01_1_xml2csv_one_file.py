# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       _1_1_xml2csv_file.py
   Description:     convert a xml file to csv
   Author:          Junda
   date:            11/6/21
-------------------------------------------------
"""
import csv
import xml.etree.ElementTree as Xet
import pandas as pd
import time
import logging
import multiprocessing as mp
from tqdm import tqdm
import argparse
from util import separate_text_code, clean_html_tags
from xml.etree.ElementTree import iterparse
from os import listdir
from os.path import isfile, join
import os
from datetime import datetime
import dateutil
from dateutil import parser
import xml.etree.ElementTree as ET
d1 = datetime(2014, 7, 1)
manager = mp.Manager()
q_to_store = manager.Queue()

def xml2csv(file_path):
    cnt = 0
    with open(file_path,'r', encoding='utf-8', errors='surrogatepass') as f:
        lines = f.readlines()
        for line in lines:
            try:
                tree = ET.fromstring(line)
                if tree.tag == "row" and int(tree.attrib['PostTypeId']) == 1:
                    cnt += 1
                    Id = tree.attrib["Id"]
                    # AcceptedAnswerId = check_nullable("AcceptedAnswerId", i)
                    CreationDate = tree.attrib["CreationDate"]
                    # Score = i.attrib["Score"]
                    # ViewCount = i.attrib["ViewCount"]
                    Body, Code = separate_text_code(tree.attrib["Body"])
                    # LastActivityDate = i.attrib["LastActivityDate"]
                    # FavoriteCount = check_nullable("FavoriteCount", i)
                    # AnswerCount = check_nullable("AnswerCount", i)
                    # CommentCount = check_nullable("CommentCount", i)
                    Title = tree.attrib["Title"]
                    Tags = clean_html_tags(tree.attrib["Tags"])
                    if Tags == "":
                        Tags = tree.attrib["Tags"]
                    Title = " ".join(Title.split())
                    Body = " ".join(Body.split())
                    if not Code == "":
                        Code = " ".join(Code.split())
                    row = [ Id,
                        # "AcceptedAnswerId": AcceptedAnswerId,
                        CreationDate,
                        # "Score": Score,
                        # "ViewCount": ViewCount,
                        Title,
                        Body,
                        Code,
                        # "LastActivityDate": LastActivityDate,
                        # "FavoriteCount": FavoriteCount,
                        # "AnswerCount": AnswerCount,
                        # "CommentCount": CommentCount,
                        Tags.replace('<', ' ').replace('>', ' ').strip().split(),
                    ]
                    # if cnt % 10000 == 0:
                    #     logging.info(cnt)
                    q_to_store.put(row)
            except Exception as e:
                logging.info(e)
                logging.info(line)
                


def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML file to CSV')
    parser.add_argument('--isFile', '-f', default='True',
                        help='specify the input format')
    parser.add_argument('--input', '-p',  type=str, default="../data/new_tagdc_1", help='input directory path')
    parser.add_argument('--output', '-o',  type=str, default="../data/tagdc_csv/new_tagdc_Posts.csv", help='output file path')
    args = parser.parse_args()
    input_path = args.input
    output_file = args.output
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    
    
    file_paths = []
    for root, dirs, files in os.walk(input_path):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    file_paths.sort()
    
    pbar = tqdm(total=len(file_paths))
    update = lambda *args: pbar.update()
    pool = mp.Pool(mp.cpu_count())
    logging.info("cpu count {}".format(mp.cpu_count()))
    start_time = time.time()
    for target_file in file_paths:
        pool.apply_async(xml2csv, args=(target_file,), callback=update)
    pool.close()
    pool.join()
    

    logging.info("Time cost: {} s".format(str(time.time()-start_time)))
    
    logging.info("Start to write csv file...")
    rows = []
    cols = ["Id", \
            "CreationDate",\
            "Title", \
            # "AcceptedAnswerId", \
            # "Score", "ViewCount", \
            "Body", "Code", \
            # "LastActivityDate","FavoriteCount", "AnswerCount","CommentCount", \
            "Tags"]
    while not q_to_store.empty():
        single_data = q_to_store.get()
        rows.append(single_data)
    logging.info("Writing csv")
    cnt = 0
    err = 0
    with open("../data/tagdc_csv/Posts.csv", "w", errors='surrogatepass') as _file:
        writer = csv.writer(_file)
        writer.writerow(cols)
        for row in rows:
            try:
                writer.writerow(row)
                cnt += 1
            except Exception as e:
                logging.info("Skip id={}".format(row[0]))
                logging.info("Error msg: {}".format(e))
                err += 1
            if cnt % 100000 == 0:
                logging.info("Processing {} row...".format(cnt))
    logging.info("final count {}".format(cnt))

if __name__ == "__main__":
    main()