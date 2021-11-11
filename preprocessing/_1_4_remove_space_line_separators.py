# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       _1_4_remove_space_line_separators.py
   Description:     remove redundant spaces, filter NA
   Author:          Junda
   date:            11/6/21
-------------------------------------------------
"""

import pandas as pd
import argparse


def main():

    parser = argparse.ArgumentParser(
        description='preprocess csv file')
    parser.add_argument('--input', '-p',  type=str,
                        default="../data/questions/Questions54.csv",
                        required=True, help='input csv file path')
    parser.add_argument('--output', '-o',  type=str,
                        default='../data/questions/Questions54FilteredNA.csv',
                        required=True, help='output csv file path')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    df = pd.read_csv(input_file)

    print(f"####### The initial size of the dataset is {df.shape[0]} ")

    # drop NA values in title, tags and body
    df = df.dropna(subset=['Title', 'Tags', 'Body']).reset_index()

    print(
        f"####### The size of the dataset after dropping NA value in title, body and tagis {df.shape[0]} ")

    for i, row in df.iterrows():
        # remove redundant spaces in title, body and code
        title = row['Title']
        title = " ".join(title.split())

        text = row['Body']
        text = " ".join(text.split())

        code = row['Code']
        if not code == "":
            code = " ".join(code.split())

        # process Tags
        tags = row['Tags']
        tags = tags.replace('<', ' ').replace('>', ' ').strip().split()

        df.at[i, "Title"] = title
        df.at[i, "Body"] = text
        df.at[i, "Code"] = code
        df.at[i, "Tags"] = tags

    # fill NA of code
    df[['Code']] = df[['Code']].fillna("")
    df.to_csv(output_file)


if __name__ == "__main__":
    main()
