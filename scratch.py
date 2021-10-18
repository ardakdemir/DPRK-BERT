import torch
from operator import attrgetter
from mlm_trainer import init_config
from transformers import (BertConfig, BertModel,
                          BertForPreTraining,
                          BertTokenizer,
                          AutoModel,
                          BertForMaskedLM,
                          AdamW, get_scheduler,
                          DataCollatorForLanguageModeling)
conf = init_config()

print(conf)
print(conf.vocab_size,conf.hidden_size)
p = "/Users/ardaakdemir/dprk_research/kr-bert-pretrained/pytorch_model_char16424_bert.bin"
p2 = "/Users/ardaakdemir/dprk_research/experiment_outputs/2021-10-17_11-26-44/best_model_weights.pkh"

for p_ in [p,p2]:
    weights = torch.load(p_,map_location="cpu")
    print(p_)
    for w,v in weights.items():
        if w.startswith("cls"):
            print(w,v.shape)
        # print(n,v.shape)
    model = BertForMaskedLM(conf)
    if p_ == p:
        model.cls.predictions.decoder = torch.nn.Linear(768,16424,bias=False)
    incompatible_keys = model.load_state_dict(weights, strict=False)
    print("Incompatible key info: {}".format(incompatible_keys))
# print(model)
# attr =  'cls.predictions.bias'
# print(getattr(model,attr).shape)
# setattr(model,attr,torch.tensor([0]))
# print(getattr(model,attr))

