import sys
sys.path.append("../bert")
from keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.metrics import  precision_score ,recall_score
import heapq
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K1
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from sklearn.metrics import f1_score
import gensim
from scipy.spatial.distance import cosine
import os
import pandas
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
from collections import defaultdict
import os
import pandas as pd
from util.data_util import get_fixed_tag_encoder
from data_structure.question import TagDCQuestion
import argparse
from modelbuild import Classifier
# 参数设置
vocab_dim = 256 # 向量维度
maxlen = 100  # 文本保留的最大长度
batch_size = 512
n_epoch = 150
input_length = 100

def topk(array_list,i):
    
    max_num_index_list = map(array_list.index, heapq.nlargest(i, array_list))
    return (list(max_num_index_list))

def labelProcess(labels):
    labs = []
    for label in labels:
        label = list(map(int,label))
        labs.append(label)
    return labs  

def getSamples(path, mlb):
    sentences = []
    labels = []
    data = pd.read_pickle(path)
    length = len(data)
    for i in range(length):
        text = data[i].get_desp().split()
        tags = data[i].get_tag()
        sentences.append(text)
        label = mlb.transform([set(tags)])
        labels.append(label)
    lab = labelProcess(labels)
    return sentences, lab
      
 
def getsim_list():
    labels = []
    for i in range(4784):
        print(i)
        labs = open("/data/lican/StackOverflowsmall/similarity/similarity%ld.txt"%(i),encoding = 'utf-8').read().split()   
        labels.append(labs)
    return labels
     

def simProcess():
    labels = getsim_list()
    labs = []
    for label in labels:
        label = list(map(float,label))
        labs.append(label)
    return labs          
 
 
 
def text2index(index_dic,sentences):
    
    #把词语转换为数字索引,比如[['中国','安徽','合肥'],['安徽财经大学','今天','天气','很好']]转换为[[1,5,30],[2,3,105,89]]
    new_sentences=[]
    for sen in sentences:
        new_sen=[]
        for word in sen:
            try:
                new_sen.append(index_dic[word])
            except:
                new_sen.append(0)
        new_sentences.append(new_sen)
    return new_sentences
  
def evaluate_dl(model, p_X_train, p_y_train, p_X_test, p_y_test):
    """
    :param p_n_symbols: word2vec训练后保留的词语的个数
    :param p_embedding_weights: 词索引与词向量对应矩阵
    :param p_X_train: 训练X
    :param p_y_train: 训练y
    :param p_X_test: 测试X
    :param p_y_test: 测试y0
    :return: 
    """
    """各层的结构和输入输出"""
     
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print ('Test score:', score)
    print ('Test accuracy:', acc)
     
    y_pred1 = model.predict(p_X_test)
    print(np.shape(y_pred1))
    
    y_prediction5 = []
    for i in range(len(y_pred1)):
        yy = []
        max_index = topk(y_pred1[i].tolist(),5)
        for p in range(len(y_pred1[i].tolist())):
            if p in max_index:
                yy.append(1)
            else:
                yy.append(0)
        y_prediction5.append(yy)
        yy = []
    y_true = np.vstack(p_y_test)
    y_pred5 = np.vstack(y_prediction5)
 
    print("top5")
    print("top5----precision_score:",precision_score(y_true, y_pred5,average = "samples"))
    print("top5----recall_score:",recall_score(y_true, y_pred5,average = 'samples'))
    #print(classification_report(y_true, y_pred5,target_names=tags)) 
    report1 = classification_report(y_true, y_pred5,output_dict=True)
    df = pandas.DataFrame(report1).transpose()
    df.to_excel('/data/lican/StackOverflowsmall/dl5.xlsx')
     
 
def collaborative_filtering():
    similaritylist = simProcess()
    scorelist = []
    for i in range(4784):
        score = scorecompute(similaritylist[i],y_train1,60)
        scorelist.append(score)
    print(scorelist)
    print(scorelist[1])
 
    for i in range(0,len(scorelist)):
 
        print(i)
        filename = "/data/lican/StackOverflowsmall/cf/Tag_numpy%d.txt"%(i+1)
        print(scorelist[i])
        with open(filename,'w',encoding = 'utf-8',errors = 'ignore') as f:
            f.write(" ".join("%s"%id for id in scorelist[i]))
 
     
    print(np.shape(scorelist))
    y_prediction5 = []
     
    for i in range(len(scorelist)):
        yy = []
        max_index = topk(scorelist[i].tolist(),5)
        for p in range(len(scorelist[i].tolist())):
            if p in max_index:
                yy.append(1)
            else:
                yy.append(0)
        y_prediction5.append(yy)
        yy = []
    y_true = np.vstack(y_test1)
    y_pred5 = np.vstack(y_prediction5)
 
    y_prediction10 = []
     
    for i in range(len(scorelist)):
        yy = []
        max_index = topk(scorelist[i].tolist(),10)
        for p in range(len(scorelist[i].tolist())):
            if p in max_index:
                yy.append(1)
            else:
                yy.append(0)
        y_prediction10.append(yy)
        yy = []
    y_pred10 = np.vstack(y_prediction10)
    print("top5")
    print("top5----precision_score:",precision_score(y_true, y_pred5,average = "samples"))
    print("top5----recall_score:",recall_score(y_true, y_pred5,average = 'samples'))
    #print(classification_report(y_true, y_pred5,target_names=tags)) 
    report1 = classification_report(y_true, y_pred5,output_dict=True)
    df = pandas.DataFrame(report1).transpose()
    df.to_excel('/data/lican/StackOverflowsmall/cf5.xlsx')
     
 
     
    print("top10")
    print("top10----precision_score:",precision_score(y_true, y_pred10,average = "samples"))
    print("top10----recall_score:",recall_score(y_true, y_pred10,average = 'samples'))
    #print(classification_report(y_true, y_pred10,target_names=tags))
    report2 = classification_report(y_true, y_pred10,output_dict=True)
    df = pandas.DataFrame(report2).transpose()
    df.to_excel('/data/lican/StackOverflowsmall/cf10.xlsx')
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_folder", default="../data/tagdc", type=str,
                        help="The direcoty of the input training data files.")
    parser.add_argument("--test_data_folder", default="../data/test", type=str,
                        help="The direcoty of the input training data files.")
    parser.add_argument("--vocab_file", default="../../data/tags/commonTags_post2vec.csv", type=str,
                        help="The tag vocab data file.")
    parser.add_argument("--maxlen", default=66, type=int)
    args = parser.parse_args()
    maxlen= args.maxlen
    input_dir = args.data_folder
    w2c_path = "../data/w2vec/"
    index_dict=pickle.load(open(w2c_path + 'w2indx.pkl','rb'))
    vec_dict = pickle.load(open(w2c_path + 'w2vec.pkl','rb'))
    
    n_words=len(index_dict.keys())
    vec_matrix=np.zeros((n_words+1,256))
    for k,i in index_dict.items():#将所有词索引与词向量一一对应
        try:
            vec_matrix[i,:]=vec_dict[k]
        except:
     
            print (k,i)
            print (vec_dict[k])
            exit(1)
    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            file_paths.append(os.path.join(root, file_name))
    
    demo_file = file_paths[0]
    mlb, num_class = get_fixed_tag_encoder(args.vocab_file)

    # for file_path in file_paths:
    # get training data
    X_train_txt, y_train1 = getSamples(demo_file,mlb)
    # get test data
    X_test1 = labels[:43052]
    y_test1 = labels[43052:]
   
    X_train = text2index(index_dict,X_train_txt)
    X_test = text2index(index_dict, X_test1)
     
    print(u"xshape ", np.shape(X_train))
    print(u"xshape ", np.shape(X_test))
    y_train=np.array(y_train1)
    y_test =np.array(y_test1)
    print(u"yshape ", np.shape(y_train))
    print(u"yshape ", np.shape(y_test))

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen,padding='post',truncating='post', value=0)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen,padding='post',truncating='post', value=0)
    print (u"shape ", np.shape(X_train))
    print (u"shape ", np.shape(X_test))
    
    print(np.shape(sentences))
    model = Classifier()
    checkpointer = ModelCheckpoint('chk_classifier.hdf5',monitor = 'val_loss', 
                        
                       verbose = 1, 
                       save_best_only = True, 
                       save_weights_only = True)
    early_stopper = EarlyStopping(monitor = 'val_loss', 
                       min_delta = 0.001, 
                       patience = 20)
    lr_reducer = ReduceLROnPlateau(monitor = 'val_loss',
                       factor = 0.1,
                       verbose = 1,
                       patience = 3,
                       min_lr = 2E-6)
 
    model.fit(p_X_train, p_y_train, batch_size=batch_size, epochs=n_epoch,validation_split=0.1,verbose=1, callbacks = [checkpointer, early_stopper, lr_reducer])
     
    # train_tagdc_dl(n_words+1, vec_matrix, X_train, y_train, X_test, y_test)

 
if __name__ == "__main__":
    main()