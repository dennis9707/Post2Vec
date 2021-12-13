#encoding = utf-8
import sys
sys.path.append("../bert")
from gensim.models import Word2Vec
import pickle
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from data_structure.question import Question, TagDCQuestion
import os
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for file_name in files:
                path = os.path.join(root, file_name)
                for data in pd.read_pickle(path):
                    yield data.get_desp()
                
 
def saveWordIndex(model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(list(model.wv.index_to_key), allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 词语的词向量
    pickle.dump(w2indx,open("../data/w2vec/w2indx.pkl", 'wb'))  # 索引字典
    print("w2indx")
    pickle.dump(w2vec,open("../data/w2vec/w2vec.pkl", 'wb'))  # 词向量字典
    print("w2vec")
    return w2indx, w2vec

def trainWord2Vec():#训练word2vec模型并存储
    target_dir = "../data/tagdc"
    sentences = MySentences(target_dir) # a memory-friendly iterator
    model=Word2Vec(sentences=sentences,vector_size=256,sg=1,min_count=1,window=5)
    model.save('../data/w2vec/word2vec.model')
    print("word2vec")
    # model=Word2Vec.load('./word2vec.model')
    saveWordIndex(model=model)



trainWord2Vec()

