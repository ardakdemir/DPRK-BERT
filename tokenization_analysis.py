import torch
from transformers import BertConfig, BertModel, BertForPreTraining, BertTokenizer, AutoModel
from unicodedata import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np
import json
import pickle
from net import SentenceClassifier
from tqdm import tqdm
import os
import re
import string
from cleaner import Cleaner
import config

UNK_TOKEN = "[UNK]"
DATA_ROOT = config.DATA_ROOT
# vocab_file_path = "kr-bert-pretrained/vocab_snu_subchar12367.pkl"
# weight_file_path = "kr-bert-pretrained/pytorch_model_subchar12367_bert.bin"
# config_file_path = "kr-bert-pretrained/bert_config_subchar12367.json"
# vocab_path= "kr-bert-pretrained/vocab_snu_subchar12367.txt"

vocab_file_path = "kr-bert-pretrained/vocab_snu_char16424.pkl"
weight_file_path = "kr-bert-pretrained/pytorch_model_char16424_bert.bin"
config_file_path = "kr-bert-pretrained/bert_config_char16424.json"
vocab_path = "kr-bert-pretrained/vocab_snu_char16424.txt"
tokenizer_krbert = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)

with open(config_file_path, "rb") as r:
    config_dict = json.load(r)
print("Loaded config", config_dict)


# weights = torch.load(weight_file_path, map_location="cpu")
#
# config = BertConfig(config_file_path)
# num_classes = 10
# sentence_classifier = SentenceClassifier(config,10)
# incompatible_keys = sentence_classifier.load_state_dict(weights,strict=False)
# print(incompatible_keys)


class Tokenizer:
    def __init__(self, tokenizer, cleaner=None,subchar=False):
        self.tokenizer = tokenizer
        self.start_token = "[CLS]"
        self.end_token = "[SEP]"
        self.subchar = subchar
        self.cleaner = cleaner

    def transform(self, string):
        if self.subchar:
            string = normalize("NFKD", string)
        if self.cleaner:
            string = self.cleaner.clean(string)
        tokens = self.tokenizer.tokenize(string)
        tokens = [self.start_token] + tokens + [self.end_token]
        token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        return token_ids, tokens

    def detokenize(self, tokens):
        string = []
        cur_token = ""
        for t in tokens[1:-1]:
            if t.startswith("##"):
                cur_token += t[2:]
            else:
                if len(cur_token) > 0:
                    string.append(cur_token)
                cur_token = t
        return " ".join(string)


def compare_tokenizations(tokenizer, dprk_sentence, rok_sentence):
    dprk_tokens = tokenizer_krbert.tokenize(dprk_sentence)
    rok_tokens = tokenizer_krbert.tokenize(rok_sentence)
    print("DPRK:", dprk_sentence, dprk_tokens)
    print("ROK:", rok_sentence, rok_tokens, "\n")


# convert a string into sub-char
def to_subchar(string):
    return normalize('NFKD', string)


def get_documents(json_file_path):
    d = json.load(open(json_file_path, "rb"))
    documents = []
    for k, v in d.items():
        t = v.get("type", "news_article")
        if t == "news_article":
            documents.append(v["article"].split("\n"))
        else:
            documents.append(v["letter"].split("\n"))
    return documents


def get_all_documents(root_folder):
    documents = []
    for x in os.listdir(root_folder):
        if not x.endswith("json"): continue
        p = os.path.join(root_folder, x)
        docs = get_documents(p)
        documents.extend(docs)
    return documents


def check_unknown_words(documents):
    save_path = os.path.join(DATA_ROOT, "sentences_with_unk.txt")
    unk_save_path = os.path.join(DATA_ROOT, "unknown_word_list.txt")
    problem_save_path = os.path.join(DATA_ROOT, "problem_data.txt")
    tokenizer = Tokenizer(tokenizer_krbert)
    cleaner = Cleaner()
    print("Checking {} documents in total.".format(len(documents)))
    unk_syllables = defaultdict(list)
    sent_count = 0
    problem_data = []
    for d in tqdm(documents, desc="Articles"):
        for sent in d:
            # sentence = documents[0][0]
            # sentence = "2018 2020일"
            sentence = cleaner.clean(sent)
            token_ids, tokens = tokenizer.transform(sentence)
            if UNK_TOKEN in tokens:
                detokens = tokenizer.detokenize(tokens)
                detoks = detokens.split(" ")
                try:
                    words = sentence.split(" ")
                    for w in words:
                        tok = tokenizer.tokenizer.tokenize(w)[0]
                        if tok == UNK_TOKEN:
                            for s in w:
                                tok = tokenizer.tokenizer.tokenize(s)[0]
                                if tok == UNK_TOKEN:
                                    unk_syllables[s].append({"word": w, "sentence": sentence})
                    sent_count += 1
                except Exception as e:
                    continue

    print("Problem data: {}\nUnknown Syllables: {}\nSentence wth unk words: {}".format(len(problem_data),
                                                                                       len(unk_syllables),
                                                                                       sent_count))
    with open(save_path, "w") as o:
        string = []
        for s,examples in unk_syllables.items():
            np.random.shuffle(examples)
            my_string = "# {}\n".format(s)
            for i in range(min(len(examples),5)):
                example = examples[i]
                my_string = my_string + "{} : {}\n".format(example['word'],example['sentence'])
            string.append(my_string)
        o.write("\n".join(string))

    with open(unk_save_path, "w") as o:
        counts = [(k, len(v)) for k, v in unk_syllables.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        s = "Token\tCount\n"
        s = s + "\n".join(["{}\t{}".format(c[0], c[1]) for c in counts])
        o.write(s)


def get_unknown_syllables(unknown_word_list_file):
    unknown_word_list = [(x.split()[0], x.split()[1]) for x in open(unknown_word_list_file, "r").read().split("\n")]
    save_path = os.path.join(DATA_ROOT, "unknown_syllables.txt")
    tokenizer = Tokenizer(tokenizer_krbert)
    cleaner = Cleaner()
    print("Checking {} documents in total.".format(len(documents)))
    unk_syl_counts = defaultdict(int)
    unk_word_list = []
    for w, c in unknown_word_list:
        for syl in w:
            if syl == " ": continue
            tok = tokenizer.tokenizer.tokenize(syl)[0]
            if tok == "[UNK]":
                unk_syl_counts[syl] += int(c)
    with open(save_path, "w") as o:
        counts = [(k, v) for k, v in unk_syl_counts.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        s = "Token\tCount\n"
        s = s + "\n".join(["{}\t{}".format(c[0], c[1]) for c in counts])
        o.write(s)


# Test getting sentences
json_file_path = "../rodong_parser/data/parsed_rodong_pages_1210_1923/parsed_2018-01-03.json"
json_root_folder = "../rodong_parser/data/parsed_rodong_pages_1210_1923"
documents = get_all_documents(json_root_folder)

check_unknown_words(documents)
# check_unknown_syllables(documents)
unk_save_path = os.path.join(DATA_ROOT, "unknown_word_list.txt")
# get_unknown_syllables(unk_save_path)
