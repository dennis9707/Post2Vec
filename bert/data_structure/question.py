# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     question.py
   Description :
   Author :        
   date：          2021/11/7 11:02 AM
-------------------------------------------------
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import ast
from nltk.tokenize.treebank import TreebankWordDetokenizer


class Question:
    # use __slots to decrease the memory usage
    __slots__ = ['qid', 'title', 'desc_text',
                 'desc_code', 'creation_date', 'tags', 'combine']

    def __init__(self, qid=None, title=None, desc_text=None, desc_code=None, creation_date=None, tags=None):
        self.qid = qid
        self.title = title
        self.desc_text = desc_text
        self.desc_code = desc_code
        self.creation_date = creation_date
        self.tags = tags
        self.combine = None

    def get_comp_by_name(self, comp_name):
        if comp_name == "qid":
            return self.qid
        if comp_name == "title":
            return TreebankWordDetokenizer().detokenize(self.title)
        if comp_name == "desc_text":
            return TreebankWordDetokenizer().detokenize(self.desc_text)
        if comp_name == "desc_code":
            return TreebankWordDetokenizer().detokenize(self.desc_code)
        if comp_name == "creation_date":
            return self.creation_date
        if comp_name == "tags":
            return self.tags
        if comp_name == "combine":
            return self.combine


class QuestionDataset(Dataset):

    def __init__(self, df, mlb, tokenizer):
        self.title = df['title']
        self.text = df['desc_text']
        self.code = df['desc_code']
        self.targets = df['clean_tags']
        self.tokenizer = tokenizer
        self.mlb = mlb

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        text = str(self.text[index])
        code = str(self.code[index])

        tokens = title + " " + text + " " + code

        inputs = self.tokenizer(
            tokens,
            None,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tags = self.targets[index]
        labels = set(ast.literal_eval(str(tags)))
        ret = self.mlb.transform([labels])
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'labels': torch.from_numpy(ret[0]).type(torch.FloatTensor)
        }


class BERTQuestionDataset(Dataset):

    def __init__(self, df, mlb, tokenizer):
        self.title = df['title']
        self.text = df['desc_text']
        self.code = df['desc_code']
        self.targets = df['clean_tags']
        self.tokenizer = tokenizer
        self.mlb = mlb

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        text = str(self.text[index])
        code = str(self.code[index])

        tokens = title + " " + text + " " + code

        inputs = self.tokenizer(
            tokens,
            None,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        tags = self.targets[index]
        labels = set(ast.literal_eval(str(tags)))
        ret = self.mlb.transform([labels])
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'labels': torch.from_numpy(ret[0]).type(torch.FloatTensor)
        }


class DistilBERTQuestionDataset(Dataset):

    def __init__(self, df, mlb, tokenizer):
        self.title = df['title']
        self.text = df['desc_text']
        self.code = df['desc_code']
        self.targets = df['clean_tags']
        self.tokenizer = tokenizer
        self.mlb = mlb

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        text = str(self.text[index])
        code = str(self.code[index])

        tokens = title + " " + text + " " + code

        inputs = self.tokenizer(
            tokens,
            None,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        tags = self.targets[index]
        labels = set(ast.literal_eval(str(tags)))
        ret = self.mlb.transform([labels])
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'labels': torch.from_numpy(ret[0]).type(torch.FloatTensor)
        }
