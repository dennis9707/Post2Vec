import ast
import csv
from datetime import datetime
from util import write_dict_to_csv
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='preprocess csv file')
    parser.add_argument('--input', '-p',  type=str,
                        default="../data/tags/20211110/Tags54.csv",
                        required=True, help='input csv file path')
    parser.add_argument('--output', '-o',  type=str,
                        default="../data/tags/20211110/Tags54_TagCount.csv",
                        required=True, help='output csv file path')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    # input
    # "id", "tags"
    # Tags.csv
    id_tags_csv_fpath = input_file

    # output
    # _1_3_tag-count-all.csv
    # "tag", "count"
    tag_cnt_csv_fapth = output_file
    tag_cnt_dict = {}
    tag_cnt_header = ["tag", "count"]

    # tag cnt
    row_num = 0
    with open(id_tags_csv_fpath, 'r') as id_tags_file:
        rd = csv.reader(id_tags_file, escapechar='\\')
        for row in rd:
            if row_num == 0:
                row_num += 1
                continue
            # tags = row[1].replace('<', ' ').replace('>', ' ').strip().split()
            row_num += 1
            tags = ast.literal_eval(row[1])
            for t in tags:
                if t in tag_cnt_dict:
                    tag_cnt_dict[t] += 1
                else:
                    tag_cnt_dict[t] = 1
            if row_num % 10000 == 0:
                print("Processing line %s" %
                      row_num, datetime.now().strftime("%H:%M:%S"))

    sort_tag_cnt = dict(sorted(tag_cnt_dict.items(),
                        key=lambda item: item[1], reverse=True))

    write_dict_to_csv(sort_tag_cnt, tag_cnt_csv_fapth, tag_cnt_header)
    print("# Tags = %s" % len(sort_tag_cnt))


if __name__ == "__main__":
    main()
