import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from model.loss import loss_fn


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class MaxPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveMaxPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class ClassifyHeader(nn.Module):
    """
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.title_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)
        self.code_pooler = AvgPooler(config)

        # self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_class)

    def forward(self, title_hidden, text_hidden, code_hidden):
        pool_title_hidden = self.title_pooler(title_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        pool_code_hidden = self.code_pooler(code_hidden)

        # concatenates the given sequence of tensors in the given dimension
        concated_hidden = torch.cat((pool_title_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, pool_code_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    

class LargeClassifyHeader(nn.Module):
    """
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config, num_class):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.title_pooler = MaxPooler(config)
        self.text_pooler = MaxPooler(config)
        self.code_pooler = MaxPooler(config)

        # self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size * 5, config.hidden_size * 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size * 3, num_class)

    def forward(self, title_hidden, text_hidden, code_hidden):
        pool_title_hidden = self.title_pooler(title_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        pool_code_hidden = self.code_pooler(code_hidden)
        diff_hidden1 = torch.abs(pool_text_hidden - pool_title_hidden)
        diff_hidden2 = torch.abs(pool_code_hidden - pool_text_hidden)

        # concatenates the given sequence of tensors in the given dimension
        concated_hidden = torch.cat((pool_title_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, pool_code_hidden), 1)
        concated_diff_hidden = torch.cat((diff_hidden1, diff_hidden2), 1)
        concated_hidden = torch.cat((concated_hidden, concated_diff_hidden), 1)
        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class TBertT(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        self.tbert = AutoModel.from_pretrained(code_bert)
        self.nbert = AutoModel.from_pretrained(code_bert)
        self.cbert = AutoModel.from_pretrained(code_bert)

        self.cls = ClassifyHeader(config, num_class=num_class)

    def forward(
            self,
            title_ids=None,
            title_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            code_ids=None,
            code_attention_mask=None,
    ):
        t_hidden = self.tbert(
            title_ids, attention_mask=title_attention_mask)[0]
        n_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]
        c_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]

        logits = self.cls(title_hidden=t_hidden,
                          text_hidden=n_hidden, code_hidden=c_hidden)
        return logits


class TBertTLarge(PreTrainedModel):
    def __init__(self, config, code_bert, num_class):
        super().__init__(config)
        self.tbert = AutoModel.from_pretrained(code_bert)
        self.nbert = AutoModel.from_pretrained(code_bert)
        self.cbert = AutoModel.from_pretrained(code_bert)

        self.cls = LargeClassifyHeader(config, num_class=num_class)

    def forward(
            self,
            title_ids=None,
            title_attention_mask=None,
            text_ids=None,
            text_attention_mask=None,
            code_ids=None,
            code_attention_mask=None,
    ):
        t_hidden = self.tbert(
            title_ids, attention_mask=title_attention_mask)[0]
        n_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]
        c_hidden = self.cbert(code_ids, attention_mask=code_attention_mask)[0]

        logits = self.cls(title_hidden=t_hidden,
                          text_hidden=n_hidden, code_hidden=c_hidden)
        return logits


class TBertSI(TBertT):
    def __init__(self, config, code_bert,num_class):
        super().__init__(config, code_bert,num_class)
        self.tbert = AutoModel.from_pretrained(code_bert)
        self.nbert = self.tbert
        self.cbert = self.tbert
        self.cls = ClassifyHeader(config, num_class=num_class)