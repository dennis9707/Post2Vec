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
d1 = datetime(2014, 7, 1)
# manager = mp.Manager()
# q_to_store = manager.Queue()

def xml2csv(i):
    if i.tag == "row":
        if i.attrib['PostTypeId'] == '1':
            Id = i.attrib["Id"]
            # AcceptedAnswerId = check_nullable("AcceptedAnswerId", i)
            CreationDate = i.attrib["CreationDate"]
            date = parser.parse(CreationDate)   
            scnt += 1
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
            if scnt % 100000 == 0:
                logging.info("Processing {} row...".format(scnt))
            # q_to_store.put(row)


def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML file to CSV')
    parser.add_argument('--isFile', '-f', default='True',
                        help='specify the input format')
    parser.add_argument('--input', '-p',  type=str, default="../data/tagdc/Posts.xml", help='input directory path')
    parser.add_argument('--output', '-o',  type=str, default="../data/tagdc_csv/Posts.csv", help='output file path')
    args = parser.parse_args()
    target_file = args.input
    output_file = args.output
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Start to process files...")
    logging.info(target_file)
    cnt = 0
    start_time = time.time()
    rows =[]
    logging.info("Processing {}".format(target_file))
    for _, elem in iterparse(target_file):
        if elem.tag == 'row':
            if int(elem.attrib['PostTypeId']) == 1:
                cnt += 1
                Id = elem.attrib["Id"]
                CreationDate = elem.attrib["CreationDate"]
                Body, Code = separate_text_code(elem.attrib["Body"])
                Title = elem.attrib["Title"]
                Tags = clean_html_tags(elem.attrib["Tags"])
                if Tags == "":
                    Tags = elem.attrib["Tags"]
                Title = " ".join(Title.split())
                Body = " ".join(Body.split())
                if not Code == "":
                    Code = " ".join(Code.split())
                row = [ Id,
                    CreationDate,
                    Title,
                    Body,
                    Code,
                    Tags.replace('<', ' ').replace('>', ' ').strip().split(),
                ]
                rows.append(row)
                if cnt % 10000 == 0:
                    logging.info(cnt)
        elem.clear()
                        
    logging.info("Final number {}".format(cnt))

    logging.info("Time cost: {} s".format(str(time.time()-start_time)))
    logging.info("Start to write csv file...")
    cols = ["Id", \
            "CreationDate",\
            "Title", \
            "Body", "Code", \
            "Tags"]
        
    logging.info("Writing csv")
    cnt = 0
    err = 0
    with open("../data/tagdc_csv/Posts_iterative.csv", "w", errors='surrogatepass') as _file:
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

if __name__ == "__main__":
    main()