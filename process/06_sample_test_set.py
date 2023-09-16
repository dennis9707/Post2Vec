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

# import requests

# def check_post_exists(post_id):
#     try:
#         response = requests.get(f"https://api.stackexchange.com/2.3/posts/{post_id}?site=stackoverflow")
#         data = response.json()
        
#         if 'items' in data and data['items']:
#             return True
#         else:
#             return False
#     except requests.exceptions.RequestException as e:
#         print(f"An error occurred: {e}")
#         return False

# # Testing the function with a post ID
# post_id = 12345678  # Replace with a real post ID
# if check_post_exists(post_id):
#     print(f"The post with ID {post_id} exists on Stack Overflow.")
# else:
#     print(f"The post with ID {post_id} does not exist on Stack Overflow.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", default="../data/processed_test")
    parser.add_argument("--out_dir", "-o", default="../data/")
    parser.add_argument("--model_type", default='microsoft/codebert-base')
    args = parser.parse_args()

    # # If folder doesn't exist, then create it.
    # import os
    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir)
    #     print("created folder : ", args.out_dir)

    # else:
    #     print(args.out_dir, "folder already exists.")
    files = get_files_paths_from_directory(args.input_dir)    
    results = []

    for file in files:
        dataset = pd.read_pickle(file)
        for question in dataset:
            qid = question.get_qid()
            title = question.get_title()
            tags = question.get_tag()
            results.append({'qid': qid, 'title': title, 'tags': tags,})

    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    print("The length of the dataframe is:", len(df))
    # Write the results to a CSV file
    df.to_csv('test-data.csv', index=False)
        

if __name__ == "__main__":
    main()
    