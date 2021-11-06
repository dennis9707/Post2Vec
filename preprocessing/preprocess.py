import time
import logging
import argparse
from typing import Counter
import multiprocessing as mp

manager = mp.Manager()
q_to_store = manager.Queue()

from tqdm import tqdm
import codeprep.api.text as cp

def identifier_split(line):
    split_data = cp.basic(line, no_str=True, no_com=True, ronin = True)
        
    q_to_store.put(" ".join(split_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", "-i")
    parser.add_argument("--out_fp", "-o")

    args = parser.parse_args()

    with open(args.input_fp, "r") as f, \
            open(args.out_fp, "w") as fout:

        logging.info("Start to process files...")
        lines = f.readlines()
        pbar = tqdm(total=len(lines))
        update = lambda *args: pbar.update()

        start_time = time.time()
        pool = mp.Pool(mp.cpu_count())
        for line in lines:
            # all_tokens.append(identifier_split(line, args.split))
            pool.apply_async(identifier_split, args=(line,), callback=update)
        pool.close()
        pool.join()
        
        logging.info("Time cost: {} s".format(str(time.time()-start_time)))
        logging.info("Start to write files...")

        while not q_to_store.empty():
            single_data = q_to_store.get()
            if len(single_data):
                fout.write(single_data + "\n")
        logging.info("Done")

    with open(args.out_fp, "r") as fo:
        lines = fo.readlines()
        vocab = Counter()
        # add unk and pad tokens
        vocab_to_keep = []

        vocab_to_keep.append("-UNK-")
        vocab_to_keep.append("-EMP-")
        # fix ids of special tokens
        for line in tqdm(lines):
            try:
                vocab.update(line.strip().split(" "))
            except:
                continue
            
        vocab_to_keep += [i[0] for i in vocab.most_common()]
        logging.info("Total # of vocab: {}".format(len(vocab)))


if name == "__main__":
    main()