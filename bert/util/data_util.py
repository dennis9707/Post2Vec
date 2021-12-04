import sys
sys.path.append("../")
sys.path.append("/usr/src/bert")
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from data_structure.question import Question, QuestionDataset
from sklearn import preprocessing
import pandas as pd


def get_tag_encoder(vocab_file):
    tab_vocab_path = vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path,keep_default_na=False)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    return mlb, len(mlb.classes_)


def load_data_to_dataset(mlb, file):
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", local_files_only=True)
    train = pd.read_pickle(file)
    training_set = QuestionDataset(train, mlb, tokenizer)
    return training_set


def get_dataloader(dataset, batch_size):
    # sampler = DistributedSampler(dataset)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             )
    return data_loader


def get_distribued_dataloader(dataset, batch_size):
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             )
    return data_loader
