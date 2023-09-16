import pandas as pd
src_path = "../new_data/final/train.csv"
for i,chunk in enumerate(pd.read_csv(src_path, chunksize=140000)):
    chunk.to_csv('../new_data/tokenize_csv/posts_chunk{}.csv'.format(i), index=False)
    