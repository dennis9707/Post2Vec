import sys
sys.path.append("../bert")
from data_structure.question import NewQuestion
import pandas as pd
import pickle
import random
import os

df = []
file_paths = []
for root, dirs, files in os.walk("../data/tagdc_filter"):
    for file_name in files:
        file_paths.append(os.path.join(root, file_name))
file_paths.sort()

for file in file_paths:
    df1 = pd.read_pickle(file)
    df.extend(df1)
    print(len(df))

# split train and test
df_train = df[:10073127]
df_test = df[10073127:]

random.shuffle(df_train)

for i in range(0, len(df_train), 40000):
    target = df_train[i:i + 40000]
    with open("../data/tagdc_train/tagdc_"+str(i)+".pkl", 'wb') as f:
        pickle.dump(target, f)

import pickle
for i in range(0, len(df_test), 40000):
    target = df_test[i:i + 40000]
    with open("../data/tagdc_test/tagdc_"+str(i)+".pkl", 'wb') as f:
        pickle.dump(target, f)