import torch.nn as nn
import torch
from transformers import (BertConfig, BertModel,
                          BertForPreTraining,
                          BertTokenizer,
                          AutoModel,
                          BertForMaskedLM,
                          AdamW, get_scheduler,
                          DataCollatorForLanguageModeling)


class SentenceClassifier(nn.Module):
    def __init__(self, config, num_classes) -> None:
        super(SentenceClassifier, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, bert_input, labels=None, output_hidden_states=False):
        bert_output = self.bert(**bert_input)
        last_hidden_states = bert_output[0]
        pooled_output = self.dropout(last_hidden_states)[:,0,:]#First hidden [CLS] token
        logits = self.classifier(pooled_output)
        return logits, bert_output

    def init_bert_weights(self, weight_file_path):
        weights = torch.load(weight_file_path, map_location="cpu")
        incompatible_keys = self.bert.load_state_dict(weights, strict=False)
        print("Incompatible keys when loading bert state dict", sum(len(k) for k in incompatible_keys))
        print(incompatible_keys)

    def init_self_weights(self, weight_file_path):
        weights = torch.load(weight_file_path, map_location="cpu")
        incompatible_keys = self.load_state_dict(weights, strict=False)
        print("Incompatible keys when loading model state dict", sum(len(k) for k in incompatible_keys))
        print(incompatible_keys)
