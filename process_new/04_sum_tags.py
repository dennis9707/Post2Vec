import numpy as np
import pandas as pd
import csv
import pandas as pd
import time
import logging
import multiprocessing as mp
from tqdm import tqdm
import argparse
import os
from util import clean_html_tags, clean_html_tags2, separate_text_code

def process_csv(file_path,output_path):
    cnt = 0
    file_name = file_path[23:-4]
    logging.info(file_name)
    questions = []
    
    df = pd.read_csv(file_path)
    import ast
    row_num = 0
    tag_cnt_dict = {}
    tag_cnt_header = ["tag", "count"]
    for index, row in df.iterrows():
        tags = ast.literal_eval(row["Tags"])
        for t in tags:
            if t in tag_cnt_dict:
                tag_cnt_dict[t] += 1
            else:
                tag_cnt_dict[t] = 1
        if row_num % 50000 == 0:
            print("Processing line %s" % row_num)
        row_num += 1
    
        
    with open('../new_data/cnt/'+file_name+'.csv', 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(tag_cnt_header)
        for key, value in tag_cnt_dict.items():
            writer.writerow([key, value])


def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML file to CSV')
    parser.add_argument('--input', '-p',  type=str, default="../new_data/cnt/", help='input directory path which stores splited csv')
    parser.add_argument('--output', '-o',  type=str, default="../new_data/cnt/", help='output file path')
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
    # pbar = tqdm(total=len(file_paths))
    # update = lambda *args: pbar.update()
    # pool = mp.Pool(mp.cpu_count())
    # logging.info("cpu count {}".format(mp.cpu_count()))
    # start_time = time.time()
    # for target_file in file_paths:
    #     process_csv(target_file,output_file)  
    data_frames = []
    # Loop through all CSV files and read each into a DataFrame
    for csv_file in file_paths:
        df = pd.read_csv(csv_file)
        data_frames.append(df)

    # Concatenate all DataFrames
    merged_df = pd.concat(data_frames)

    # Group by 'tag' column and sum the 'count' column
    result_df = merged_df.groupby('tag')['count'].sum().reset_index()
    result_df = result_df.sort_values(by='count', ascending=False)

    # # Save the result to a new CSV file
    # result_df.to_csv('result.csv', index=False)

    # Save the result to a new CSV file
    result_df.to_csv('../new_data/cnt/result-sort.csv', index=False)

    
    # logging.info("Time cost: {} s".format(str(time.time()-start_time)))

if __name__ == "__main__":
    main()