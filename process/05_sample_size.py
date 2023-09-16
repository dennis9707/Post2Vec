import sys
sys.path.append("../bert")
sys.path.append("../..")
# import multiprocessing as mp
from typing import Counter
import argparse
import logging
import time
# from tqdm import tqdm
from data_structure.question import Question, NewQuestion
import pandas as pd
from util.util import get_files_paths_from_directory
# from tqdm import tqdm
import time
from transformers import AutoTokenizer
import random

import requests

def check_post_exists(post_id):
    try:
        response = requests.get(f"https://api.stackexchange.com/2.3/posts/{post_id}?site=stackoverflow")
        data = response.json()
        
        if 'items' in data and data['items']:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return False

# # Testing the function with a post ID
# post_id = 12345678  # Replace with a real post ID
# if check_post_exists(post_id):
#     print(f"The post with ID {post_id} exists on Stack Overflow.")
# else:
#     print(f"The post with ID {post_id} does not exist on Stack Overflow.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", default="../data/processed_train")
    parser.add_argument("--out_dir", "-o", default="../data/")
    parser.add_argument("--model_type", default='microsoft/codebert-base')
    parser.add_argument("--title_max", default=100)
    parser.add_argument("--text_max", default=512)
    parser.add_argument("--code_max", default=512)
    args = parser.parse_args()

    # # If folder doesn't exist, then create it.
    # import os
    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir)
    #     print("created folder : ", args.out_dir)

    # else:
    #     print(args.out_dir, "folder already exists.")
    files = get_files_paths_from_directory(args.input_dir)
    sampled_files = random.sample(files, 30)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    print(len(sampled_files))
    
    results = []
    skip_first = True

    for file in sampled_files:
        print(file)
        dataset = pd.read_pickle(file)
        print(len(dataset))
        if skip_first:
            skip_first = False
            continue
        for question in dataset:
            qid = question.get_qid()
            print(qid)
            a = check_post_exists(int(qid))
            print(a)
            if not a:
                title = question.get_title()
                text = question.get_text()
                code = question.get_code()
                date = question.get_creation_date()
                tags = question.get_tag()

                results.append({'qid': qid, 'title': title, 'text': text, 'code': code, 'tags': tags, 'date': date})
                print(len(results))                
                break

    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    print("The length of the dataframe is:", len(df))
    # Write the results to a CSV file
    df.to_csv('results1.csv', index=False)
        

if __name__ == "__main__":
    main()