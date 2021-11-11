# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       _1_1_xml2csv_file.py
   Description:     convert a xml file to csv
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
from util import separate_text_code, clean_html_tags

manager = mp.Manager()
q_to_store = manager.Queue()


def xml2csv(i):
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
        q_to_store.put(row)


def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML file to CSV')
    parser.add_argument('--isFile', '-f', default='True',
                        help='specify the input format')
    parser.add_argument('--input', '-p',  type=str,
                        required=True, help='input file path')
    parser.add_argument('--output', '-o',  type=str,
                        required=True, help='output file path')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    logging.info("Start to process files...")
    xmlparse = Xet.parse(input_file)
    root = xmlparse.getroot()
    length = len(root.findall('row'))
    pbar = tqdm(total=length)
    update = lambda *args: pbar.update()
    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())

    for i in root:
        pool.apply_async(xml2csv, args=(i,), callback=update)
    pool.close()
    pool.join()

    logging.info("Time cost: {} s".format(str(time.time()-start_time)))
    logging.info("Start to write csv file...")
    rows = []
    cols = ["Id", \
            # "AcceptedAnswerId", \
            "CreationDate",
            # "Score", "ViewCount", \
            "Body", "Code", \
            # "LastActivityDate","FavoriteCount", "AnswerCount","CommentCount", \
            "Title", "Tags"]
    while not q_to_store.empty():
        single_data = q_to_store.get()
        rows.append(single_data)
    # Writing dataframe to csv
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output_file)
    logging.info("Done")


if __name__ == "__main__":
    main()
