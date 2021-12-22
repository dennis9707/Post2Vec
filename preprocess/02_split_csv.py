import pandas as pd

for i,chunk in enumerate(pd.read_csv('../data/tagdc_csv./csv', chunksize=100000)):
    chunk.to_csv('chunk{}.csv'.format(i), index=False)