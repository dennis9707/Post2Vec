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
        labels = set(ast.literal_eval(tags))
        ret = self.mlb.transform([labels])

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.from_numpy(ret[0]).type(torch.FloatTensor)
        }
