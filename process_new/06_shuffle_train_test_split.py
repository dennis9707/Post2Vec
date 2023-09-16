import numpy as np
import pandas as pd
import csv
import time
import logging
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML file to CSV')
    parser.add_argument('--input', '-p',  type=str, default="../new_data/clean_data/", help='input directory path which stores splited csv')
    args = parser.parse_args()
    input_path = args.input
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    
    # get the file paths
    file_paths = []
    for root, dirs, files in os.walk(input_path):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    file_paths.sort()


    dfs = []
    cnt = 0
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            cnt += 1
            if cnt % 10 == 0:
                logging.info(file_path)
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

        # Concatenate all dataframes
    merged_df = pd.concat(dfs)
    # Separate the last 100,000 rows and the rest
    last_100k_samples = merged_df.iloc[-100000:]
    rest_of_samples = merged_df.iloc[:-100000]
    rest_of_samples = rest_of_samples.sample(frac=1).reset_index(drop=True)

    rest_of_samples.to_csv('../new_data/final/rest_of_samples.csv', index=False)
    last_100k_samples.to_csv('../new_data/final/last_100k_samples.csv', index=False)


if __name__ == "__main__":
    main()