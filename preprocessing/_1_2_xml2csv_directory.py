# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       _1_2_xml2csv_directory.py
   Description:     convert xml files into csv files
   Author:          Junda
   date:            11/6/21
-------------------------------------------------
"""

import xml.etree.ElementTree as Xet
import pandas as pd
import time
import logging
import multiprocessing as mp
from tqdm import tqdm
import argparse
from util import separate_text_code, clean_html_tags, clean_html_tags2, remove_symbols, check_nullable
import os
manager = mp.Manager()
q_to_store = manager.Queue()


def xml2csv(file):
    rows = []
    xmlparse = Xet.parse(file)
    root = xmlparse.getroot()
    for i in root:
        if i.attrib['PostTypeId'] == '1':
            Id = i.attrib["Id"]
            # AcceptedAnswerId = check_nullable("AcceptedAnswerId", i)
            CreationDate = i.attrib["CreationDate"]
            # Score = i.attrib["Score"]
            # ViewCount = i.attrib["ViewCount"]
            Body, Code = separate_text_code(i.attrib["Body"])
            # LastActivityDate = i.attrib["LastActivityDate"]
            # FavoriteCount = check_nullable("FavoriteCount", i)
            # AnswerCount = check_nullable("AnswerCount", i)
            # CommentCount = check_nullable("CommentCount", i)
            Title = i.attrib["Title"]
            Tags = clean_html_tags(i.attrib["Tags"])
            if Tags == "":
                Tags = i.attrib["Tags"]

            row = {"Id": Id,
                   # "AcceptedAnswerId": AcceptedAnswerId,
                   "CreationDate": CreationDate,
                   # "Score": Score,
                   # "ViewCount": ViewCount,
                   "Body": Body,
                   "Code": Code,
                   # "LastActivityDate": LastActivityDate,
                   # "FavoriteCount": FavoriteCount,
                   # "AnswerCount": AnswerCount,
                   # "CommentCount": CommentCount,
                   "Title": Title,
                   "Tags": Tags,
                   }
            rows.append(row)
    q_to_store.put(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML files to CSV')
    parser.add_argument('--input', '-p',  type=str,
                        required=True, help='input directory path')
    parser.add_argument('--output', '-o',  type=str,
                        required=True, help='output directory path')
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    logging.basicConfig(level=logging.INFO)

    logging.info("Start to process files...")
    start_time = time.time()

    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    pbar = tqdm(total=length)
    update = lambda *args: pbar.update()
    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    file_paths = file_paths[:2]
    length = len(file_paths)
    for file in file_paths:
        pool.apply_async(xml2csv, args=(file,), callback=update)
    pool.close()
    pool.join()
    logging.info("Time cost: {} s".format(str(time.time()-start_time)))
    logging.info("Start to write csv file...")
    cols = ["Id", \
            # "AcceptedAnswerId", \
            "CreationDate",
            # "Score", "ViewCount", \
            "Body", "Code", \
            # "LastActivityDate","FavoriteCount", "AnswerCount","CommentCount", \
            "Title", "Tags"]
    index = 1
    while not q_to_store.empty():
        single_data = q_to_store.get()
        # Writing dataframe to csv
        df = pd.DataFrame(single_data, columns=cols)
        docstring = output_dir + "Question{}.csv".format(index)
        df.to_csv(docstring)
        index += 1
    logging.info("Done")


if __name__ == "__main__":
    main()
