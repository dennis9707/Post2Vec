"""
Split the target file into chunks, with chunk size equal to 50000
"""

import pandas as pd

for i,chunk in enumerate(pd.read_csv("../data/tagdc_csv/stackoverflow_large.csv", chunksize=50000)):
    chunk.to_csv('../data/tagdc_csv_split/chunk{}.csv'.format(i), index=False)