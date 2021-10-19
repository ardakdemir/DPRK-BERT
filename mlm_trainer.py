from cleaner import Cleaner
import utils
from collections import defaultdict
from tokenizer import Tokenizer
from datasets import load_dataset
import json
import torch
import math
import random
import logging
from torch.utils.data import DataLoader
import torch.nn as nn
from unicodedata import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import json
import time
# from net import SentenceClassifier
from tqdm import tqdm
from parse_args import parse_args
import config as config_file
import os
import datetime as dt
from plotter import BasicPlotter, joint_bar_plot

from transformers import (BertConfig, BertModel,
                          BertForPreTraining,
                          BertTokenizer,
                          AutoModel,
                          BertForMaskedLM,
                          AdamW, get_scheduler,
                          DataCollatorForLanguageModeling)

logging.basicConfig(filename=config_file.LOG_FILE_PATH, level=logging.DEBUG)
# logger = logging.getLogger(__name__)

vocab_file_path = "../kr-bert-pretrained/vocab_snu_char16424.pkl"
weight_file_path = "../kr-bert-pretrained/pytorch_model_char16424_bert.bin"
config_file_path = "../kr-bert-pretrained/bert_config_char16424.json"
vocab_path = "../kr-bert-pretrained/vocab_snu_char16424.txt"

KR_BERT_MODEL_NAME = "pytorch_model_char16424_bert.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_exp_name(prefix=None):
    day, time = dt.datetime.now().isoformat().split("T")
    suffix = "_".join([day, "-".join(time.split(".")[0].split(":"))])
    if prefix is None:
        return suffix
    else:
        return "_".join([prefix, suffix])


def init_config(config_name=None):
    # Config
    if not config_name:
        with open(config_file.CONFIG_FILE_PATH, "rb") as r:
            config_dict = json.load(r)
        print("Loaded config", config_dict)
        config = BertConfig(**config_dict)
        print(config)
    else:
        config = BertConfig.from_pretrained(config_name)
        print(config)
    return config


def init_mlm_model(config, weight_file_path=None, from_pretrained=False):
    if not from_pretrained:
        if weight_file_path is None:
            weight_file_path = config_file.WEIGHT_FILE_PATH
        weights = torch.load(weight_file_path, map_location="cpu")
        model = BertForMaskedLM(config)
        model_name = os.path.split(weight_file_path)[-1]
        if model_name == KR_BERT_MODEL_NAME:
            print("Changing cls decoder to not have bias!!")
            model.cls.predictions.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        incompatible_keys = model.load_state_dict(weights, strict=False)
        # print(model)
        print("Incompatible keys when loading", incompatible_keys)
        return model
    else:
        model = BertForMaskedLM.from_pretrained(weight_file_path)
        print(model)
        return model


def get_corrects(batch, preds):
    labels = batch["labels"]
    mask_count = torch.sum(labels != -100).detach().cpu().item()
    print("{} masked tokens in this batch".format(mask_count))
    correct_dict = {}
    for k, p in preds.items():
        # print("Label shape {} pred shape {}".format(labels.shape, p.shape))
        indx = torch.argmax(p, dim=2)
        corrects = torch.sum(indx == labels).detach().cpu().item()
        # print("{}/{} masked tokens are correct for {} ".format(corrects, mask_count, k))
        correct_dict[k] = corrects
    return correct_dict, mask_count


def analyze_mlm_predictions(tokenizer, batch, preds, topN=10):
    """

    :param tokenizer: Tokenizer used to convert tokens into IDs
    :param batch: input to the model
    :param output: output of each model
    :return:
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    softmax_layer = nn.Softmax(dim=0)
    pred_info = []
    corrects = defaultdict(int)
    ranks = defaultdict(list)  # For Mean Reciprocal Rank (MRR)
    number_of_masks = []
    for sent_index, (input, label) in enumerate(zip(input_ids, labels)):
        tokens = tokenizer.convert_ids_to_tokens(input)
        if not "[MASK]" in tokens:
            continue
        final_ind = tokens.index("[SEP]")
        sentence = tokenizer.convert_tokens_to_string(tokens[1:final_ind])
        my_info = {"sentence_with_masks": sentence}
        raw_tokens = []
        mask_count = 0
        for t, l in zip(tokens[1:final_ind], label[1:final_ind]):
            if t == "[MASK]":
                raw_tokens.append(tokenizer.convert_ids_to_tokens([l])[0])
                mask_count += 1
            else:
                raw_tokens.append(t)
        number_of_masks.append(mask_count)
        raw_sentence = tokenizer.convert_tokens_to_string(raw_tokens)
        # print("sentence_with_masks: {}\nraw_sentence: {}".format(sentence, raw_sentence))
        my_info["raw_sentence"] = raw_sentence
        my_info['predictions'] = []
        for index, (token, lab) in enumerate(zip(tokens, label)):
            if token != "[MASK]":
                continue
            correct_word = tokenizer.convert_ids_to_tokens([lab])[0]
            data = {"mask_index": index, "masked_word": correct_word, "model_data": {}}
            for k, pred in preds.items():
                pred = pred[sent_index]
                sorted, indices = torch.sort(softmax_layer(pred[index]), descending=True)
                token_probs = [(i, v) for v, i in zip(sorted.detach().cpu().numpy(), indices.detach().cpu().numpy())]
                # token_probs.sort(key=lambda x: x[1], reverse=True)
                top_n_preds = tokenizer.convert_ids_to_tokens([x[0] for x in token_probs[:topN]])
                top_n_probs = [x[1].item() for x in token_probs[:topN]]
                correct_token_rank = -1
                for i, tok_prob in enumerate(token_probs):
                    t, p = tok_prob
                    if t == lab:
                        correct_token_rank = i + 1  # 0 indexed
                        correct_prob = float(p.item())
                        break
                # print("Model name: {} correct word: {} correct_prob {} correct_rank {} top_n_preds {} ".format(k,
                #                                                                                                correct_word,
                #                                                                                                correct_prob,
                #                                                                                                correct_token_rank,
                #                                                                                                top_n_preds))
                model_data = {"correct": correct_token_rank == 1,
                              "correct_prob": correct_prob,
                              "top_n_preds": top_n_preds,
                              "correct_rank": correct_token_rank,
                              "top_n_probs": top_n_probs}
                if correct_token_rank == 1:  # Top prediction
                    corrects[k] += 1
                ranks[k].append(correct_token_rank)
                data["model_data"][k] = model_data
            my_info['predictions'].append(data)
        pred_info.append(my_info)
    return pred_info, corrects, ranks, np.sum(number_of_masks)


def init_tokenizer(tokenizer_name=None):
    if tokenizer_name is None:
        tokenizer = BertTokenizer(config_file.VOCAB_PATH, do_lower_case=False)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
    cleaner = Cleaner()
    tokenizer = Tokenizer(tokenizer, cleaner)
    return tokenizer


def tokenize_function(examples, tokenizer, max_seq_length=512, text_column_name="data"):
    # Remove empty lines
    examples = [
        line for example in examples for line in example[text_column_name].split("\n") if
        len(line) > 0 and not line.isspace()
    ]

    return [tokenizer.tokenize(
        example,
        {"padding": True,
         "truncation": True,
         "max_length": max_seq_length,
         # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
         # receives the `special_tokens_mask`.
         "return_special_tokens_mask": True}) for example in examples]


def train():
    args = parse_args()
    config = init_config()
    model = init_mlm_model(config)
    tokenizer = init_tokenizer()

    experiment_folder = os.path.join(config_file.OUTPUT_FOLDER, get_exp_name())
    plot_save_folder = os.path.join(experiment_folder, "plots")
    basic_plotter = BasicPlotter(plot_save_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model_save_path = os.path.join(experiment_folder, "best_model_weights.pkh")

    # Tokenize dataset
    tokenized_datasets = {}
    for s in ["train", "validation"]:
        examples = raw_datasets[s]
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
        args.num_train_epochs = math.max(10, math.ceil(args.max_train_steps / num_update_steps_per_epoch))

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
    model.to(device)
    min_perplexity = 1e6
    for epoch in range(args.num_train_epochs):
        model.train()
        train_losses = []
        epoch_begin = time.time()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            logging.info("Loss: {}".format(loss.item()))
            # print("Loss item: {}".format(loss.item()))
            train_losses.append(loss.item())
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        epoch_end = time.time()
        train_epoch_time = round(epoch_end - epoch_begin, 3)
        basic_plotter.send_metrics({"train_loss": np.mean(train_losses), "train_epoch_time": train_epoch_time})

        model.eval()
        losses = []
        eval_begin = time.time()
        for step, batch in enumerate(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss)
        losses = torch.stack(losses, dim=0)
        losses = losses[: len(eval_dataset)]
        eval_end = time.time()
        eval_time = round(eval_end - eval_begin, 3)
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        basic_plotter.send_metrics({"validation_perplexity": perplexity, "validation_time": eval_time})
        # print(f"epoch {epoch}: perplexity: {perplexity}")
        if perplexity < min_perplexity:
            print("Saving best model with average perplexity of {} to: {}".format(perplexity, model_save_path))
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, model_save_path)


def evaluate_single_model(model, dataloader, tokenizer=None, break_after=2):
    model.eval()
    losses = []
    eval_begin = time.time()
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
        if tokenizer:
            analyze_mlm_prediction(tokenizer.tokenizer, batch, outputs)
        loss = outputs.loss
        losses.append(loss)
        if step > break_after:
            break
    losses = torch.stack(losses, dim=0)
    eval_end = time.time()
    eval_time = round(eval_end - eval_begin, 3)
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    print("Average Perplexity: {}".format(perplexity))
    return perplexity


def evaluate_multiple_models_mlm_wrapper(model_path_dict, dataset_path, repeat=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = ""
    results = defaultdict(list)
    config = init_config()
    tokenizer = init_tokenizer()
    raw_datasets = load_dataset("json", data_files={"test": dataset_path}, field="data")
    eval_dataset = tokenize_function(raw_datasets["test"], tokenizer)
    args = parse_args()
    break_after = args.validation_steps
    analyze_preds = args.analyze_preds
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer.tokenizer,
                                                    mlm_probability=args.mlm_probability)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                 batch_size=args.per_device_eval_batch_size)
    models = {}
    for k, model_dict in model_path_dict.items():
        print("Loading {}...".format(k))
        model_path = model_dict["model_name"]
        max_seq_length = args.max_seq_length
        config_name = model_dict["config_name"]
        model = init_mlm_model(config, weight_file_path=model_path, from_pretrained=config_name is not None)
        model.to(device)
        models[k] = model
    all_results = {}
    all_pred_info_dicts = {}
    for r in range(repeat):
        print("Starting repeat: {}".format(r))
        results, pred_info_dict = evaluate_multiple_models_mlm(models, eval_dataloader, tokenizer,
                                                               break_after=break_after,metric_only = not analyze_preds)
        all_results[r] = results
        all_pred_info_dicts[r] = pred_info_dict
    return all_results, all_pred_info_dicts


def evaluate_multiple_models_mlm(models, dataloader, tokenizer, break_after=10, metric_only=True):
    """
        For each test sentence:
            - Store top 10 predictions for the [MASK] tokens of each model with probabilities
            - Store probabilities and the RANK of the correct token for each model
    :param models:  dict -> E.g., {"KR-BERT":kr-bert-model, "DPRK-BERT":dprk-bert-model}
    :param dataloader:
    :param tokenizer:
    :param break_after:
    :return:
    """
    print("Device {}".format(device))
    for k, model in models.items():
        model.eval()
        model.to(device)
    losses = defaultdict(list)
    corrects = defaultdict(int)
    ranks = defaultdict(list)
    total_examples = 0
    eval_begin = time.time()
    prediction_info_dict = []
    results = {}
    progress_bar = tqdm(range(break_after), desc="Batches")
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        num_sents = len(batch["input_ids"])
        preds = {}

        model_input_begin = time.time()
        for k, model in models.items():
            with torch.no_grad():
                outputs = model(**batch)
            preds[k] = outputs.logits
            losses[k].append(round(outputs.loss.detach().cpu().item(),3))
        model_input_end = time.time()
        model_input_time = round(model_input_end - model_input_begin, 3)
        pred_info, ranks = None, None
        analysis_begin = time.time()
        if not metric_only:
            pred_info, batch_corrects, batch_ranks, total_masks = analyze_mlm_predictions(tokenizer.tokenizer, batch,
                                                                                          preds)
        else:
            batch_corrects,total_masks = get_corrects(batch, preds)
        analysis_end = time.time()
        analysis_time = round(analysis_end - analysis_begin, 3)
        # print("model input time", model_input_time, " analysis time ", analysis_time)
        progress_bar.update(1)
        for k, c in batch_corrects.items():
            corrects[k] += c
            if ranks is not None:
                ranks[k].extend(batch_ranks[k])
        total_examples += total_masks
        if pred_info is not None:
            prediction_info_dict.extend(pred_info)
        if step > break_after:
            break
    for k in models:
        perplexity = math.exp(np.mean(losses[k]))
        results[k] = {"perplexity": perplexity,
                      "accuracy": 100*round(corrects[k] / total_examples, 3)}
        if not metric_only:
            my_ranks = ranks[k]
            mrr = np.mean([1 / x for x in my_ranks])
            results[k]["mean_reciprocal_rank"] = round(mrr, 3)
    eval_end = time.time()
    eval_time = round(eval_end - eval_begin, 3)
    return results, prediction_info_dict


def evaluate_samemodels(model_paths, dataset_path, repeat=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = defaultdict(list)
    config = init_config()
    tokenizer = init_tokenizer()
    raw_datasets = load_dataset("json", data_files={"test": dataset_path}, field="data")
    eval_dataset = tokenize_function(raw_datasets["test"], tokenizer)
    args = parse_args()
    for i in range(repeat):
        print("Evaluation step {}...".format(i + 1))
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer.tokenizer,
                                                        mlm_probability=args.mlm_probability)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                     batch_size=args.per_device_eval_batch_size)
        for k, model_dict in model_paths.items():
            print("Evaluating {}...".format(k))
            model_path = model_dict["model_name"]
            max_seq_length = args.max_seq_length
            config_name = model_dict["config_name"]
            model = init_mlm_model(config, weight_file_path=model_path, from_pretrained=config_name is not None)
            model.to(device)
            model.eval()
            losses = []
            eval_begin = time.time()
            perplexity = evaluate_single_model(model, eval_dataloader, tokenizer)
            results[k].append(perplexity)
    print("Finished all evaluations...")
    print("Results", results)
    return results


def evaluate_model_perplexity(model_paths, dataset_path, repeat=5):
    """

    :param model_paths: E.g., {"KR-BERT":<path-to-krbert-model>,"DPRK-BERT":<path-to-dprkbert-model>}
    :param dataset_path:
    :param repeat:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = defaultdict(list)
    for i in range(repeat):
        print("Evaluation step {}...".format(i + 1))
        for k, model_dict in model_paths.items():
            print("Evaluating {}...".format(k))
            config_name = model_dict["config_name"]
            tokenizer = model_dict["tokenizer"]
            model_path = model_dict["model_name"]
            config = init_config(config_name)
            tokenizer = init_tokenizer(tokenizer)
            raw_datasets = load_dataset("json", data_files={"test": dataset_path}, field="data")
            args = parse_args()
            eval_dataset = tokenize_function(raw_datasets["test"], tokenizer)

            max_seq_length = args.max_seq_length

            model = init_mlm_model(config, weight_file_path=model_path, from_pretrained=config_name is not None)
            model.to(device)
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer.tokenizer,
                                                            mlm_probability=args.mlm_probability)
            eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                         batch_size=args.per_device_eval_batch_size)
            model.eval()
            losses = []
            eval_begin = time.time()
            for step, batch in enumerate(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(loss)
                if step > 5:
                    break
            losses = torch.stack(losses, dim=0)
            losses = losses[: len(eval_dataset)]
            eval_end = time.time()
            eval_time = round(eval_end - eval_begin, 3)
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")
            print("Average Perplexity: {}".format(perplexity))
            results[k].append(perplexity)
    print("Finished all evaluations...")
    print("Results", results)
    return results


def evaluate():
    # model_paths = {"KR-BERT": "../kr-bert-pretrained/pytorch_model_char16424_bert.bin",
    #                "DPRK-BERT": "../experiment_outputs/2021-10-17_02-36-11/best_model_weights.pkh"}
    # model_paths = {"mBERT": {"model_name":"bert-base-multilingual-cased",
    #                          "tokenizer":"bert-base-multilingual-cased",
    #                          "config_name":"bert-base-multilingual-cased"},
    #                "DPRK-BERT": {"model_name":"../experiment_outputs/2021-10-17_02-36-11/best_model_weights.pkh",
    #                              "tokenizer":None,"config_name":None}}
    args = parse_args()
    model_paths = {"KR-BERT": {
        "model_name": "../kr-bert-pretrained/pytorch_model_char16424_bert.bin",
        "tokenizer": None,
        "config_name": None},
        "DPRK-BERT": {"model_name": "../experiment_outputs/2021-10-17_02-36-11/best_model_weights.pkh",
                      "tokenizer": None, "config_name": None}}
    # dataset_path = "../dprk-bert-data/rodong_mlm_training_data/validation.json"
    dataset_path = "../dprk-bert-data/new_year_mlm_data/train.json"

    prefix = "comparison"
    exp_name = get_exp_name(prefix)
    y_label = "perplexity"
    print(exp_name)
    save_folder = os.path.join("../experiment_outputs", exp_name)
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    predinfo_save_path = os.path.join(save_folder, "prediction_info_dict.json")
    result_save_path = os.path.join(save_folder, "results.json")
    experiment_detail_save_path = os.path.join(save_folder, "experiment_details.json")

    experiment_details = {"dataset_path":dataset_path,"model_paths":model_paths}
    # results = evaluate_model_perplexity(model_paths, dataset_path)
    all_results, all_prediction_info_dict = evaluate_multiple_models_mlm_wrapper(model_paths, dataset_path,
                                                                                 repeat=args.mlm_eval_repeat)
    print(all_results)
    print("Experiment name ",exp_name)
    with open(predinfo_save_path, "w") as o:
        json.dump(all_prediction_info_dict, o)
    with open(result_save_path, "w") as o:
        json.dump(all_results, o)
    with open(experiment_detail_save_path, "w") as o:
        json.dump(experiment_details, o)
    repeat_keys = list(all_results.keys())
    model_keys = list(all_results[repeat_keys[0]].keys())
    all_metrics = list(all_results[repeat_keys[0]][model_keys[0]].keys())
    print("All metrics", all_metrics)
    for metric in all_metrics:
        results = {m: [all_results[x][m][metric] for x in repeat_keys] for m in model_keys}
        plot_save_path = os.path.join(save_folder, "{}.png".format(metric))
        joint_bar_plot(results, plot_save_path, y_label=metric.title())


def main():
    args = parse_args()
    mode = args.mode
    if mode == "train":
        train()
    elif mode == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()
