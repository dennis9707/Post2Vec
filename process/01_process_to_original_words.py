import sys
sys.path.append("../bert")
sys.path.append("../..")
from util.util import get_files_paths_from_directory
import pandas as pd
from data_structure.question import Question, NewQuestion
from tqdm import tqdm
import time
import logging
import argparse
from typing import Counter
import multiprocessing as mp
import os


def process_file(file):
    out_dir = "../data/processed_test/"
    file_name = file[12:]
    q_list = list()
    data = pd.read_pickle(file)
    length = len(data)
    for i in range(length):
        qid = data[i].get_comp_by_name("qid")
        title = data[i].get_comp_by_name("title")
        text = data[i].get_comp_by_name("desc_text")
        code = data[i].get_comp_by_name("desc_code")
        date = data[i].get_comp_by_name("creation_date")
        tags = data[i].get_comp_by_name("tags")
        q_list.append(NewQuestion(qid, title, text, code, date, tags))
    # write to pickle
    import pickle
    with open(out_dir+file_name, 'wb') as f:
        pickle.dump(q_list, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", default="../data/test")
    parser.add_argument("--out_dir", "-o", default="../data/processed_train")
    args = parser.parse_args()

    # If folder doesn't exist, then create it.
    if not args.out_dir:
        os.makedirs(args.out_dir)
        print("created folder : ", args.out_dir)

    else:
        print(args.out_dir, "folder already exists.")
    files = get_files_paths_from_directory(args.input_dir)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    pbar = tqdm(total=len(files))
    update = lambda *args: pbar.update()

    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    for file in files:
        pool.apply_async(process_file, args=(file, ), callback=update)
    pool.close()
    pool.join()

    logging.info("Time cost: {} s".format(str(time.time()-start_time)))
    logging.info("Start to write files...")
    logging.info("Done")


if __name__ == "__main__":
    main()
