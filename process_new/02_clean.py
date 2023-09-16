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
    file_name = file_path[24:-4]
    logging.info(file_name)
    questions = []
    
    input_file = pd.read_csv(file_path)
    for index, row in input_file.iterrows():
        if row['PostTypeId'] == 1:
            try:
                Id = row["Id"],
                CreationDate = row["CreationDate"],
                Title =  ' '.join(clean_html_tags(row["Title"]).split())
                Body, Code = separate_text_code(row["Body"])
                Body = " ".join(Body.split())
                if not Code == "":
                    Code = " ".join(Code.split())
                Tags = clean_html_tags(row["Tags"])
                if Tags == "":
                    Tags = row["Tags"]
                question = {"Id": Id,
                            "CreationDate": CreationDate,
                            "Title": Title,
                            "Body": Body,
                            "Code": Code,
                            "Tags": Tags.replace('<', ' ').replace('>', ' ').strip().split(),
                            }
                questions.append(question)
                cnt += 1
                
                if (cnt + 1) % 50000 == 0:
                    logging.info("count {}".format(cnt))
                    # keys = questions[0].keys()

                    # with open('../new_data/posts2023/'+file_name+'.csv', 'w', errors='surrogatepass') as output_file:
                    #     dict_writer = csv.DictWriter(output_file, keys)
                    #     dict_writer.writeheader()
                    #     dict_writer.writerows(questions)
            except Exception as e:
                logging.info(e)
                logging.info(row["Id"])
    logging.info("{} Total number of questions {} ".format(file_name, cnt))
    logging.info("Start to write csv file...")
    keys = questions[0].keys()

    with open('../new_data/post2023/'+file_name+'.csv', 'w', errors='surrogatepass') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(questions)


def main():
    parser = argparse.ArgumentParser(
        description='Change format from XML file to CSV')
    parser.add_argument('--input', '-p',  type=str, default="../new_data/split_csv", help='input directory path which stores splited csv')
    parser.add_argument('--output', '-o',  type=str, default="../new_data/posts2023/", help='output file path')
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
    # for target_file in file_paths:
    #     process_csv(target_file,output_file)

    for target_file in file_paths:
        pool.apply_async(process_csv, args=(target_file,output_file), callback=update)
    pool.close()
    pool.join()
    
    logging.info("Time cost: {} s".format(str(time.time()-start_time)))

if __name__ == "__main__":
    main()