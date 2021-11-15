import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from transformers import AutoModel
from transformers import BertModel


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
