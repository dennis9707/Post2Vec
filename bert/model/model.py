import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from transformers import AutoModel
from transformers import BertModel
from transformers import DistilBertModel, DistilBertConfig
from transformers import AlbertTokenizer, AlbertModel

# Initializing a DistilBERT configuration


class CodeBERTModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CodeBERTModel, self).__init__()
        self.codebert_model = AutoModel.from_pretrained(
            "microsoft/codebert-base", return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, ids, mask):
        output = self.codebert_model(ids, attention_mask=mask)['pooler_output']
        output_dropout = self.dropout(output)
        x = self.fc1(output_dropout)
        logit = self.fc2(x)
        return logit


class BertForMultiLable(torch.nn.Module):
    def __init__(self, num_classes):
        super(BertForMultiLable, self).__init__()
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)
        return logits


class DistillBertForMultiLable(torch.nn.Module):
    def __init__(self, num_classes):
        super(DistillBertForMultiLable, self).__init__()
        self.distillbert = AutoModel.from_pretrained(
            'distilbert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distillbert(
            input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)
        return logits


class RobertaForMultiLable(torch.nn.Module):
    def __init__(self, num_classes):
        super(RobertaForMultiLable, self).__init__()
        self.distillbert = AutoModel.from_pretrained(
            'distilroberta-base', return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distillbert(
            input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)
        return logits


class AlbertForMultiLable(torch.nn.Module):
    def __init__(self, num_classes):
        super(AlbertForMultiLable, self).__init__()
        self.bert = AlbertModel.from_pretrained(
            'albert-base-v2', return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)
        return logits
