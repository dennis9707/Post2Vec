import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from transformers import AutoModel


class CodeBERTModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CodeBERTModel, self).__init__()
        self.codebert_model = AutoModel.from_pretrained(
            'microsoft/codebert-base', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, ids):
        print(ids)
        output = self.codebert_model(ids)['pooler_output']
        output_dropout = self.dropout(output)
        x = self.fc1(output_dropout)
        logit = self.fc2(x)
        output = sigmoid(logit)
        return output
