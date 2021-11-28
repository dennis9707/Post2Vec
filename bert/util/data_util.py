import pandas as pd
from sklearn import preprocessing
 
def get_tag_encoder(vocab_file):
    tab_vocab_path = vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    return mlb, len(mlb.classes_)