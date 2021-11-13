# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       _2_1_identify_rare_tags_for_file.py
   Description:
   Author:          Junda
   date:            11/6/21
-------------------------------------------------
"""

import os
import csv
from datetime import date
from util import get_date, load_tag_cnt, write_list_to_csv


def identify_rare_tags(tag_dict, rare_tags_fpath, commom_tags_fpath, ts):
    rare_tags = []
    common_tags = []
    for t in tag_dict:
        if tag_dict[t] < ts:
            rare_tags.append(t)
        else:
            common_tags.append(t)
    header = ["tag"]
    write_list_to_csv(rare_tags, rare_tags_fpath, header)
    write_list_to_csv(common_tags, commom_tags_fpath, header)
    print("#rare tags : %s" % len(rare_tags) + '\n')


if __name__ == '__main__':
    ################### Path Setting ##########################
    dataset_dir = "../data/tags/" + get_date
    # ts
    ts = 1000
    ts_dir = dataset_dir + os.sep + "ts%s" % ts

    if not os.path.exists(ts_dir):
        os.mkdir(ts_dir)

    # Input:
    tag_cnt_csv_fapth = dataset_dir + os.sep + "Tags54_TagCount_Sorted.csv"

    # Output:
    rare_tags_fpath = ts_dir + os.sep + "_2_1_rareTags.csv"
    common_tags_fpath = ts_dir + os.sep + "_2_1_commonTags.csv"

    ################### Path Setting ##########################

    # <tag_cnt>
    tag_cnt = load_tag_cnt(tag_cnt_csv_fapth)
    # get rare tags [rare tags]
    identify_rare_tags(tag_cnt, rare_tags_fpath, common_tags_fpath, ts)
