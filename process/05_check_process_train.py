
import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
from data_structure.question import NewQuestion, QuestionDataset
from util.util import get_files_paths_from_directory
import pandas as pd
path = "../data/processed_train"
files = get_files_paths_from_directory(path)
train = pd.read_pickle(files[0])
q = train[0]
print(q.get_text())