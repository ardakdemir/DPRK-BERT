import os
from sa_reader import read_data_from_folder
from net import SentenceClassifier
from mlm_trainer import get_exp_name, init_tokenizer
import config as config_file
from plotter import BasicPlotter, joint_bar_plot
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
import argparse
import utils
from transformers import (BertConfig, BertModel,
                          BertForPreTraining,
                          BertTokenizer,
                          AutoModel,
                          BertForMaskedLM,
                          AdamW, get_scheduler,
                          DataCollatorForLanguageModeling)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on sentiment analysis task")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="../dprk-bert-data/dprk_sentiment_data_2612",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help="Path to store the files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix of the path to store the files",
    )
    parser.add_argument(
        "--bert_weight_file_path",
        type=str,
        default=None,
        help="path to bert model weights.",
    )

    parser.add_argument(
        "--weight_file_path",
        type=str,
        default="../DPRK-BERT-model/dprk-model/best_model_weights.pkh",
        help="path to best model weights.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--steps_per_epoch", type=int, default=1e9, help="Number of steps per epoch.")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    args = parser.parse_args()
    return args


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


def init_sa_model(num_classes, config, weight_file_path=None, bert_weight_file_path=None):
    sentence_classifier = SentenceClassifier(config, num_classes)
    if weight_file_path:
        sentence_classifier.init_self_weights(weight_file_path)
    elif bert_weight_file_path:
        sentence_classifier.init_bert_weights(bert_weight_file_path)
    print(sentence_classifier)

    return sentence_classifier


def collate_function(batch, pad_id=0):
    pad_map = {'attention_mask': 0, "token_type_ids": 0, "input_ids": pad_id, "label": -100}
    all_keys = list(batch[0].keys())
    max_length = 0
    prepared_batch = []
    for b in batch:
        max_length = max(max_length, len(b["input_ids"]))
    for b in batch:
        my_length = len(b["input_ids"])
        for k, pad_index in pad_map.items():
            if k != "label":
                b[k] = b[k] + [pad_index] * (max_length - my_length)
        prepared_batch.append(b)
    return {k: torch.tensor([p[k] for p in prepared_batch]) for k in pad_map}


def prepare_dataset(examples, tokenizer, max_seq_length=512, text_column_name="data"):
    # Remove empty lines
    sentences = [
        line for example in examples for line in example[text_column_name].split("\n") if
        len(line) > 0 and not line.isspace()
    ]
    tokenized_data = []
    label_vocab = {}
    for e, sentence in zip(examples, sentences):
        tokenized_input = tokenizer.tokenize(sentence,
                                             {"padding": True,
                                              "truncation": True,
                                              "max_length": max_seq_length})
        l = e["tag"]
        if l not in label_vocab:
            label_vocab[l] = len(label_vocab)
        e["label"] = label_vocab[l]
        tokenized_input.update(e)
        tokenized_data.append(tokenized_input)
    return tokenized_data, label_vocab


def train(model, data_dict, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_weights = None
    best_acc = 0
    best_loss = 1e9

    train_summary = {}

    label_map = {"pos": args.label_vocab["positive-sentence"],
                 "neg": args.label_vocab["negative-sentence"]}

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()

    model.train()
    model.to(device)
    all_losses = defaultdict(list)
    num_epochs = args.num_train_epochs
    for n in range(num_epochs):
        epoch_results = {}
        for k, data in data_dict.items():
            losses = []
            preds = []
            truths = []
            print("Running on ", k)
            if k == "test":
                model.eval()
            if k == "train":
                model.train()
            step = 0
            for i, batch in tqdm(enumerate(data), desc="Iteration"):
                batch = {k: v.to(device) for k, v in batch.items()}
                bert_input = {k: batch[k] for k in ['attention_mask', "token_type_ids", "input_ids"]}

                if k == "test":
                    with torch.no_grad():
                        logits, bert_output = model(bert_input)
                else:
                    logits, bert_output = model(bert_input)

                label = batch["label"]

                # print(logits.shape,label.shape,label)
                batch_preds = torch.argmax(logits, axis=1).detach().cpu().numpy().tolist()
                batch_truths = label.detach().cpu().numpy().tolist()
                preds.extend(batch_preds)
                truths.extend(batch_truths)

                loss = criterion(logits, label)
                if (i + 1) % 50 == 0:
                    print(np.mean(losses))
                losses.append(loss.item())
                if k == "train":
                    loss.backward()
                    optimizer.step()
                step += 1
                if step > args.steps_per_epoch:
                    break
            mean_loss = np.mean(losses)
            all_losses[k].append(mean_loss)
            result = utils.measure_classification_accuracy(preds, truths, label_map=label_map)
            result_wo_indices = {k: v for k, v in result.items() if k != "wrong_inds"}
            print("Results for {}: ".format(k), result_wo_indices)
            if k == "test" and mean_loss < best_loss:
                print("Found best model at {} epoch".format(n))
                print("Result",result_wo_indices)
                best_loss = mean_loss
                best_model_weights = model.state_dict()
            epoch_results[k] = result

        train_summary[f"epoch_{n}"] = epoch_results
    train_summary["all_losses"] = all_losses

    return best_model_weights, train_summary


def main():
    args = parse_args()
    save_folder = args.save_folder
    prefix = args.prefix
    # init save_folders
    if save_folder is None:
        save_folder = get_exp_name(prefix)
    experiment_folder = os.path.join(config_file.OUTPUT_FOLDER, save_folder)

    plot_save_folder = os.path.join(experiment_folder, "plots")
    basic_plotter = BasicPlotter(plot_save_folder)

    # read data
    dataset_folder = args.dataset_folder
    file_names = ["train", "test"]
    data_dict = read_data_from_folder(dataset_folder, file_names)

    # init model
    num_classes = 2
    config = init_config()
    weight_file_path = args.weight_file_path
    bert_weight_file_path = args.bert_weight_file_path
    sa_model = init_sa_model(num_classes, config, weight_file_path=weight_file_path,
                             bert_weight_file_path=bert_weight_file_path)

    # init tokenizer
    tokenizer, tokenizer_name = init_tokenizer(from_pretrained=False)
    print("Tokenizer", tokenizer_name)

    # init dataset
    tokenized_datasets = {}
    data_loaders = {}
    label_vocab = None

    for s in file_names:
        examples = data_dict[s]
        tokenized_samples, label_vocab = prepare_dataset(examples, tokenizer)
        tokenized_datasets[s] = tokenized_samples
    for s in file_names:
        dataset = tokenized_datasets[s]
        # print(f"Sample from {s}", len(dataset[0]), len(dataset[1]), dataset[0])
        dl = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_function)
        data_loaders[s] = dl

    # train
    args.label_vocab = label_vocab
    best_model, train_summary = train(sa_model, data_loaders, args)
    # test

    # print results
    s_p = os.path.join(experiment_folder, "label_vocab.json")
    with open(s_p, "w") as o:
        json.dump(label_vocab, o)


if __name__ == "__main__":
    main()
