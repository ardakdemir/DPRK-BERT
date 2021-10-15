from cleaner import Cleaner
import utils
from tokenizer import Tokenizer
from datasets import load_dataset
import json
import torch
import math
import random
import logging
from torch.utils.data import DataLoader
from unicodedata import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import json
from net import SentenceClassifier
from tqdm import tqdm
from parse_args import parse_args
import config as config_file
import os
from transformers import (BertConfig, BertModel,
                          BertForPreTraining,
                          BertTokenizer,
                          AutoModel,
                          BertForMaskedLM,
                          AdamW,get_scheduler,
                          DataCollatorForLanguageModeling)

logging.basicConfig(filename=config_file.LOG_FILE_PATH, level=logging.DEBUG)
# logger = logging.getLogger(__name__)

vocab_file_path = "../kr-bert-pretrained/vocab_snu_char16424.pkl"
weight_file_path = "../kr-bert-pretrained/pytorch_model_char16424_bert.bin"
config_file_path = "../kr-bert-pretrained/bert_config_char16424.json"
vocab_path = "../kr-bert-pretrained/vocab_snu_char16424.txt"



def init_config():
    # Config
    with open(config_file.CONFIG_FILE_PATH, "rb") as r:
        config_dict = json.load(r)
    print("Loaded config", config_dict)
    config = BertConfig(**config_dict)
    print(config)
    return config

def init_mlm_model(config):
    weights = torch.load(config_file.WEIGHT_FILE_PATH, map_location="cpu")
    model = BertForMaskedLM(config)
    incompatible_keys = model.load_state_dict(weights, strict=False)
    print("Incompatible keys when loading",incompatible_keys)
    return model

def init_tokenizer():
    tokenizer_krbert = BertTokenizer(config_file.VOCAB_PATH, do_lower_case=False)
    cleaner = Cleaner()
    tokenizer = Tokenizer(tokenizer_krbert, cleaner)
    return tokenizer

def tokenize_function(examples, tokenizer, text_column_name="data"):
    # Remove empty lines
    examples = [
        line for example in examples for line in example[text_column_name].split("\n") if
        len(line) > 0 and not line.isspace()
    ]
    return [tokenizer.tokenize(
        example,
        {"padding": True,
         "truncation": True,
         "max_length": args.max_seq_length,
         # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
         # receives the `special_tokens_mask`.
         "return_special_tokens_mask": True}) for example in examples]

def train():
    args = parse_args()
    config = init_config()
    model = init_mlm_model(config)
    tokenizer = init_tokenizer()

    # Dataset
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = "json"
    raw_datasets = load_dataset(extension, data_files=data_files, field="data")

    max_seq_length = args.max_seq_length
    column_names = raw_datasets["train"].column_names
    column_name = "data"

    # Tokenize dataset
    tokenized_datasets = {}
    for s in ["train", "validation"]:
        examples = [raw_datasets[s][i] for i in range(5)]
        tokenized_datasets[s] = tokenize_function(examples, tokenizer)

        # print(tokenizer_output)
        print("Number of sentences: {}".format(len(tokenized_datasets[s])))
        #
        # with open(os.path.join(config_file.RODONG_MLM_ROOT, "tokenized_{}.json".format(s)), "w") as o:
        #     json.dump({"tokenized_data": tokenizer_output}, o)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]


    print("MLM with probability: {}".format(args.mlm_probability))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer.tokenizer, mlm_probability=args.mlm_probability)

    # Data loaders
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.max(10,math.ceil(args.max_train_steps / num_update_steps_per_epoch))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logging.info("Starting training")

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    print("Starting the training...")
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # print(batch)
            shape = batch["input_ids"].shape
            print("Input id shape: {}".format(shape))
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            logging.info("Loss: {}".format(loss.item()))
            print("Loss item: {}".format(loss.item()))
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.item())

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        print(f"epoch {epoch}: perplexity: {perplexity}")

def main():
    train()

if __name__ == "__main__":
    main()