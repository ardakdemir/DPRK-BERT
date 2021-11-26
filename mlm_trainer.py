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
from timer import Timer
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
        config = BertConfig(**config_dict)
        print("Bert Config", config)
    else:
        config = BertConfig.from_pretrained(config_name)
        print("Bert Config", config)
    return config


def init_tokenizer(tokenizer_name_or_path=None, from_pretrained=True):
    if tokenizer_name_or_path is None and not from_pretrained:
        tokenizer = BertTokenizer(config_file.VOCAB_PATH, do_lower_case=False)
        tokenizer_name_or_path = "default"
    else:
        if from_pretrained:
            tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path, do_lower_case=False)
        else:
            tokenizer = BertTokenizer(tokenizer_name_or_path, do_lower_case=False)
    cleaner = Cleaner()
    tokenizer = Tokenizer(tokenizer, cleaner)
    return tokenizer, tokenizer_name_or_path


def cl_regularization(hidden_states_1, hidden_states_2, args):
    hidden_1 = hidden_states_1[-2]
    hidden_2 = hidden_states_2[-2]
    weight = 0.1
    shape = hidden_states_2[-2].shape  ## Last layer before mask head
    dim = 2
    if len(shape) == 2:  # means no batching weird!!
        dim = 1

    return torch.mean(torch.norm(hidden_1 - hidden_2, dim=dim)) * weight


def init_krbert():
    # Get config
    with open(config_file.CONFIG_FILE_PATH, "rb") as r:
        config_dict = json.load(r)
        bert_config = BertConfig(**config_dict)
    weights = torch.load(config_file.WEIGHT_FILE_PATH, map_location="cpu")
    krbert = BertForMaskedLM(bert_config)
    incom_keys = krbert.load_state_dict(weights, strict=False)
    print("Missing keys ", len(incom_keys.missing_keys), "Unexpected keys ", len(incom_keys.unexpected_keys))
    return krbert


def init_mlm_model(config, weight_file_path=None, from_pretrained=False, from_scratch=False):
    if not from_pretrained:
        model = BertForMaskedLM(config)
        if from_scratch:
            print("Using randomly initialized BERT model for mlm....")
            return model
        else:
            print("Loading weights from a BERT checkpoint for mlm...")
            model_name = os.path.split(weight_file_path)[-1]
            if weight_file_path is None:
                weight_file_path = config_file.WEIGHT_FILE_PATH
            weights = torch.load(weight_file_path, map_location="cpu")
            if model_name == KR_BERT_MODEL_NAME:
                print("Changing cls decoder to not have bias!!")
                model.cls.predictions.decoder = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            incompatible_keys = model.load_state_dict(weights, strict=False)
            # print(model)
            print("Incompatible keys when loading", incompatible_keys)
            print("Loaded model from file", model)
            return model
    else:
        model = BertForMaskedLM.from_pretrained(weight_file_path)
        print("Loaded model from pretrained", model)
        return model


def init_mlm_models_from_dict(model_dicts):
    models = {}
    for k, model_dict in model_dicts.items():
        model_name = model_dict["model_name"]
        config = model_dict.get("config_name", None)
        tokenizer = model_dict.get("tokenizer", None)
        from_pretrained = model_dict.get("from_pretrained", False)
        bert_config = init_config(config_name=config)
        model = init_mlm_model(bert_config, weight_file_path=model_name, from_pretrained=from_pretrained)
        tokenizer, tokenizer_name = init_tokenizer(tokenizer_name_or_path=tokenizer, from_pretrained=from_pretrained)
        models[k] = {"config": bert_config,
                     "model": model,
                     "tokenizer": tokenizer,
                     "tokenizer_name": tokenizer_name}
    return models


def get_corrects(batch, preds):
    labels = batch["labels"]
    mask_count = torch.sum(labels != -100).detach().cpu().item()
    # print("{} masked tokens in this batch".format(mask_count))
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
    train_from_scratch = args.train_from_scratch
    turnoff_clr = args.turnoff_clr

    config_path = args.config_name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else config_file.VOCAB_PATH

    bert_config = init_config(config_path)
    weight_path = config_file.WEIGHT_FILE_PATH
    model = init_mlm_model(bert_config, weight_path, from_scratch=train_from_scratch)
    tokenizer, tokenizer_name = init_tokenizer(tokenizer_name_or_path=tokenizer_name,
                                               from_pretrained=not train_from_scratch)

    # Later generalize this step
    if not turnoff_clr:
        print("Initializing the kr-bert model for regularization")
        initial_bert_model = init_krbert()
        initial_bert_model.eval()  # Always in eval mode? is this a wrong idea?

    prefix = "train"
    save_folder = args.save_folder
    if save_folder is None:
        save_folder = get_exp_name(prefix)
    experiment_folder = os.path.join(config_file.OUTPUT_FOLDER, save_folder)

    plot_save_folder = os.path.join(experiment_folder, "plots")
    basic_plotter = BasicPlotter(plot_save_folder)
    timer = Timer()
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
    print("Train size: {}\tEval size: {}".format(len(train_dataloader), len(eval_dataloader)))
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

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    print("Starting the training...")
    model.to(device)
    if not turnoff_clr:
        initial_bert_model.to(device)
    min_perplexity = 1e6
    cl_regularization_terms = []
    all_train_losses = []
    all_eval_losses = []
    for epoch in range(args.num_train_epochs):
        model.train()
        train_losses = []
        epoch_begin = time.time()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            loss = outputs.loss
            if not turnoff_clr:
                with torch.no_grad():
                    batch.update({"output_hidden_states": True})
                    initial_bert_model_output = timer("Bert output time", initial_bert_model, (), batch)

                cl_regularization_term = timer("CL regularization", cl_regularization, (outputs.hidden_states,
                                                                                        initial_bert_model_output.hidden_states,
                                                                                        args))
                cl_regularization_terms.append(cl_regularization_term.item())
                if step % args.regularizer_append_steps == 0 or step == len(train_dataloader) - 1:
                    basic_plotter.send_metrics(
                        {"cl_regularizer_term": np.mean(cl_regularization_terms[-step:])})  # from last update
                if args.with_cl_regularization:
                    loss = loss + cl_regularization_term

            loss = loss / args.gradient_accumulation_steps
            train_losses.append(loss.item())
            timer("Loss backward", loss.backward, ())
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                timer("Optimizer step", optimizer.step,())
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            if completed_steps >= args.steps_per_epoch:
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
            min_perplexity = perplexity
    basic_plotter.store_json()


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


def evaluate_multiple_models_mlm_wrapper(model_dicts, dataset_path, repeat=5):
    args = parse_args()
    break_after = args.validation_steps
    analyze_preds = args.analyze_preds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = ""
    results = defaultdict(list)
    raw_datasets = load_dataset("json", data_files={"test": dataset_path}, field="data")
    dataloaders = {}
    models = init_mlm_models_from_dict(model_dicts)
    for k, model_dict in models.items():
        tokenizer_name = model_dict["tokenizer_name"]
        if tokenizer_name in dataloaders:
            continue
        tokenizer = model_dict["tokenizer"]
        eval_dataset = tokenize_function(raw_datasets["test"], tokenizer)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer.tokenizer,
                                                        mlm_probability=args.mlm_probability)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                     batch_size=args.per_device_eval_batch_size)
        dataloaders[tokenizer_name] = eval_dataloader
    all_results = {}
    all_pred_info_dicts = {}
    for r in range(repeat):
        print("Starting repeat: {}".format(r))
        results, pred_info_dict = mlm_evaluate_multiple(models, dataloaders, break_after=break_after,
                                                        metric_only=not analyze_preds)
        all_results[r] = results
        all_pred_info_dicts[r] = pred_info_dict
    return all_results, all_pred_info_dicts


def mlm_evaluate_multiple(model_dicts, dataloaders, break_after=10, metric_only=True):
    """
       MLM Evaluation for multiple models
    """
    print("Device {}".format(device))
    for k, model_dict in model_dicts.items():
        model = model_dict["model"]
        model.eval()
        model.to(device)
    losses = defaultdict(list)
    corrects = defaultdict(int)
    ranks = defaultdict(list)
    total_examples = defaultdict(int)
    eval_begin = time.time()
    prediction_info_dict = []
    results = {}
    progress_bar = tqdm(range(break_after * len(model_dicts)), desc="Batches for all models")
    for k, model_dict in model_dicts.items():
        model = model_dict["model"]
        tokenizer_name = model_dicts[k]["tokenizer_name"]
        dataloader = dataloaders[tokenizer_name]
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            num_sents = len(batch["input_ids"])
            preds = {}
            model_input_begin = time.time()
            model = model_dict["model"]
            with torch.no_grad():
                outputs = model(**batch)
            preds[k] = outputs.logits
            losses[k].append(round(outputs.loss.detach().cpu().item(), 3))
            model_input_end = time.time()
            model_input_time = round(model_input_end - model_input_begin, 3)
            pred_info, batch_ranks = None, None
            analysis_begin = time.time()
            if not metric_only:
                pred_info, batch_corrects, batch_ranks, total_masks = analyze_mlm_predictions(tokenizer.tokenizer,
                                                                                              batch,
                                                                                              preds)
            else:
                batch_corrects, total_masks = get_corrects(batch, preds)
            analysis_end = time.time()
            analysis_time = round(analysis_end - analysis_begin, 3)
            # print("model input time", model_input_time, " analysis time ", analysis_time)
            progress_bar.update(1)
            for k, c in batch_corrects.items():
                corrects[k] += c
                if batch_ranks is not None:
                    ranks[k].extend(batch_ranks[k])
            total_examples[k] += total_masks
            if pred_info is not None:
                prediction_info_dict.extend(pred_info)
            if step > break_after:
                break
    for k in model_dicts:
        perplexity = math.exp(np.mean(losses[k]))
        results[k] = {"perplexity": round(perplexity, 3),
                      "accuracy": round(100 * corrects[k] / total_examples[k], 3)}
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
    tokenizer, tokenizer_name = init_tokenizer()
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
            tokenizer, tokenizer_name = init_tokenizer(tokenizer)
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
    args = parse_args()
    dprk_model_path = "../experiment_outputs/2021-10-17_02-36-11/best_model_weights.pkh"
    tokenizer_name = args.tokenizer_name
    config_name = args.config_name
    if args.model_name_or_path is not None:
        dprk_model_path = args.model_name_or_path

    model_dict = {"KR-BERT": {
        "model_name": "../kr-bert-pretrained/pytorch_model_char16424_bert.bin",
        "tokenizer": None,
        "config_name": None},
        "DPRK-BERT": {"model_name": dprk_model_path,
                      "tokenizer": tokenizer_name, "config_name": config_name},
        "KR-BERT-MEDIUM": {"model_name": "snunlp/KR-Medium",
                           "tokenizer": "snunlp/KR-Medium",
                           "config_name": "snunlp/KR-Medium",
                           "from_pretrained": True},
        "mBERT": {"model_name": "bert-base-multilingual-cased",
                  "tokenizer": "bert-base-multilingual-cased",
                  "config_name": "bert-base-multilingual-cased",
                  "from_pretrained": True}
    }
    dataset_path = args.validation_file
    prefix = "comparison"

    exp_name = get_exp_name(prefix)
    save_folder = args.save_folder if args.save_folder is not None else exp_name
    save_folder = os.path.join(config_file.OUTPUT_FOLDER, save_folder)
    print("Save folder", save_folder)
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    predinfo_save_path = os.path.join(save_folder, "prediction_info_dict.json")
    result_save_path = os.path.join(save_folder, "results.json")
    experiment_detail_save_path = os.path.join(save_folder, "experiment_details.json")

    experiment_details = {"dataset_path": dataset_path, "model_dict": model_dict}
    # results = evaluate_model_perplexity(model_paths, dataset_path)
    all_results, all_prediction_info_dict = evaluate_multiple_models_mlm_wrapper(model_dict, dataset_path,
                                                                                 repeat=args.mlm_eval_repeat)
    print(all_results)
    print("Experiment name ", exp_name)
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
