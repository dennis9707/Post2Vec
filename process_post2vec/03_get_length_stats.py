import multiprocessing as mp
from typing import Counter
import argparse
import logging
import time
from tqdm import tqdm
from data_structure.question import Question, NewQuestion
import pandas as pd
from util.util import get_files_paths_from_directory
import sys
sys.path.append("../bert")
sys.path.append("../..")


def process_file(file):
    out_dir = "../data/processed_train/"
    file_name = file[14:]
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
    parser.add_argument("--input_dir", "-i", default="../data/train")
    parser.add_argument("--out_dir", "-o", default="../data/processed_train")

    args = parser.parse_args()
    files = get_files_paths_from_directory(args.input_dir)
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Start to process files...")
    pbar = tqdm(total=len(files))
    update = lambda *args: pbar.update()

    start_time = time.time()
    pool = mp.Pool(mp.cpu_count())
    for file in files:
        pool.apply_async(process_file, args=(file, ), callback=update)
        # result.get()
    pool.close()
    pool.join()

    logging.info("Time cost: {} s".format(str(time.time()-start_time)))
    logging.info("Start to write files...")
    logging.info("Done")


if __name__ == "__main__":
    main()
