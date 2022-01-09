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
import subprocess
from itertools import product
from transformers import (BertConfig, BertModel,
                          BertForPreTraining,
                          BertTokenizer,
                          AutoModel,
                          BertForMaskedLM,
                          AdamW, get_scheduler,
                          DataCollatorForLanguageModeling)

model_name_map = {"mbert": "bert-base-multilingual-cased",
                  "bert_en": "bert-base-cased"}


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
        "--from_transformers",
        action="store_true",
        default=False,
        help="If passed, initialize a huggingface model.",
    )
    parser.add_argument(
        "--transformer_model_name",
        type=str,
        default=None,
        help="Transformer model name",
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
        # default=None,
        help="path to best model weights.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--steps_per_epoch", type=int, default=1e9, help="Number of steps per epoch.")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--dropout_rate", type=float, default=0.15, help="dropout_rate")

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


def write_wrong_examples(wrong_examples, tokenizer, wrong_save_path):
    s = []
    s.append("Sentence\tprediction\tlabel")
    wrong_example_list = []
    for e in wrong_examples:
        input_ids, label_dict = e
        pred = label_dict["pred"]
        label = label_dict["label"]
        sep_token_id = tokenizer.tokenizer.sep_token_id
        tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids[:input_ids.index(sep_token_id)])
        detokenized = tokenizer.detokenize(tokens)
        s.append("\t".join([detokenized, pred, label]))

        d = {"detokenized": detokenized,
             "pred": pred,
             "label": label}
        wrong_example_list.append(d)

    with open(wrong_save_path, "w", encoding="utf-8") as o:
        o.write("\n".join(s))

    return wrong_example_list


def init_sa_model(num_classes, config, weight_file_path=None, bert_weight_file_path=None, from_pretrained=False,
                  model_name=None):
    sentence_classifier = SentenceClassifier(config, num_classes, from_transformers=from_pretrained,
                                             model_name=model_name)
    if not from_pretrained:
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


def prepare_dataset(examples, tokenizer, label_vocab={}, max_seq_length=512, text_column_name="data"):
    # Remove empty lines
    sentences = [
        line for example in examples for line in example[text_column_name].split("\n") if
        len(line) > 0 and not line.isspace()
    ]
    tokenized_data = []
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
    best_result = None

    train_summary = {}

    label_map = {"pos": args.label_vocab["positive-sentence"],
                 "neg": args.label_vocab["negative-sentence"]}

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()

    model.train()
    model.to(device)
    all_losses = defaultdict(list)
    num_epochs = args.num_train_epochs
    test_results = defaultdict(list)
    train_results = defaultdict(list)
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
            examples = []
            for i, batch in tqdm(enumerate(data), desc="Iteration"):
                examples.extend(batch["input_ids"].numpy().tolist())
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
                    optimizer.zero_grad()
                step += 1
                if step > args.steps_per_epoch:
                    break
            mean_loss = np.mean(losses)
            all_losses[k].append(mean_loss)
            result = utils.measure_classification_accuracy(preds, truths, examples, label_map=label_map)
            result_wo_indices = {k: v for k, v in result.items() if k != "wrong_examples"}
            acc = result["acc"]
            print("Results for {}: ".format(k), result_wo_indices)
            if k == "test" and acc > best_acc:
                print("Found best model at {} epoch with acc: {}".format(n, acc))
                print("Result", result_wo_indices)
                best_acc = acc
                best_model_weights = model.state_dict()
                best_result = result
            epoch_results[k] = result_wo_indices
            for key, v in result_wo_indices.items():
                if k == "test":
                    test_results[key].append(v)
                else:
                    train_results[key].append(v)
        train_summary[f"epoch_{n}"] = epoch_results
    train_summary["all_losses"] = all_losses
    train_summary["best_test_acc"] = best_acc
    train_summary["best_result"] = best_result
    train_summary["train_results"] = train_results
    train_summary["test_results"] = test_results

    return best_model_weights, train_summary


def main():
    args = parse_args()
    save_folder = args.save_folder
    config_name = None
    model_name = None
    from_transformers = args.from_transformers
    if from_transformers:
        assert args.transformer_model_name, "Transformer model name must be defined when initializing from transformers."
        model_name = model_name_map[args.transformer_model_name]
        config_name = model_name
        model_name = model_name
    prefix = args.prefix

    dropout = np.arange(0, 0.55, 0.05)
    learning_rate = [5e-5, 5e-4, 1e-4, 1e-3, 5e-3]
    learning_rate = [args.learning_rate]
    dropout = [args.dropout_rate]
    best_model = None
    best_metric = 0

    # init save_folders
    if save_folder is None:
        save_folder = get_exp_name(prefix)

    for index, c in tqdm(enumerate(product(dropout, learning_rate)), desc="Hyperparam search"):
        d_r, l_r = c
        args.dropout_rate = d_r
        args.learning_rate = l_r
        print("Running for dropout: {}\tlearning_rate: {}".format(d_r, l_r))

        experiment_folder = os.path.join(config_file.OUTPUT_FOLDER, save_folder, str(index))
        if not os.path.exists(experiment_folder):
            os.makedirs(experiment_folder)
        plot_save_folder = os.path.join(experiment_folder, "plots")
        basic_plotter = BasicPlotter(plot_save_folder)

        # read data
        dataset_folder = args.dataset_folder
        file_names = ["train", "test"]
        data_dict = read_data_from_folder(dataset_folder, file_names)

        # init model
        num_classes = 2
        config = init_config(config_name)
        weight_file_path = args.weight_file_path
        dropout_rate = args.dropout_rate
        config.hidden_dropout_prob = dropout_rate
        bert_weight_file_path = args.bert_weight_file_path
        sa_model = init_sa_model(num_classes, config, weight_file_path=weight_file_path,
                                 bert_weight_file_path=bert_weight_file_path,
                                 from_pretrained=from_transformers,
                                 model_name=model_name)

        # init tokenizer
        tokenizer, tokenizer_name = init_tokenizer(model_name, from_pretrained=from_transformers)
        print("Tokenizer", tokenizer_name)

        # init dataset
        tokenized_datasets = {}
        data_loaders = {}
        label_vocab = None

        for s in file_names:
            examples = data_dict[s]
        train_examples = data_dict["train"]
        tokenized_samples, label_vocab = prepare_dataset(train_examples, tokenizer)
        tokenized_datasets["train"] = tokenized_samples

        test_examples = data_dict["test"]
        tokenized_samples, label_vocab = prepare_dataset(test_examples, tokenizer, label_vocab=label_vocab)
        tokenized_datasets["test"] = tokenized_samples

        for s in file_names:
            dataset = tokenized_datasets[s]
            # print(f"Sample from {s}", len(dataset[0]), len(dataset[1]), dataset[0])
            dl = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_function)
            data_loaders[s] = dl

        # train-test
        args.label_vocab = label_vocab
        best_model, train_summary = train(sa_model, data_loaders, args)
        basic_plotter.send_metrics({"test_loss": train_summary["all_losses"]["test"],
                                    "train_loss": train_summary["all_losses"]["train"]})

        for k in train_summary["test_results"]:
            basic_plotter.send_metrics({f"test_{k}": train_summary["test_results"][k],
                                        f"train_{k}": train_summary["train_results"][k]})

        # print results
        s_p = os.path.join(experiment_folder, "label_vocab.json")
        with open(s_p, "w") as o:
            json.dump(label_vocab, o)

        s_p = os.path.join(experiment_folder, "args.json")
        with open(s_p, "w") as o:
            json.dump(vars(args), o)

        model_save_path = os.path.join(experiment_folder, "best_model_weights.pkh")
        torch.save(best_model, model_save_path)

        # write wrong examples
        wrong_examples = train_summary["best_result"]["wrong_examples"]
        wrong_save_path = os.path.join(experiment_folder, "wrong_examples.txt")
        wrong_example_list = write_wrong_examples(wrong_examples, tokenizer, wrong_save_path)

        # save results summary
        s_p = os.path.join(experiment_folder, "train_summary.json")
        with open(s_p, "w") as o:
            train_summary["wrong_examples"] = wrong_example_list
            json.dump(train_summary, o)

        # copy best model
        test_acc = train_summary["best_test_acc"]
        if test_acc > best_metric:
            best_metric = test_acc
            print("Best model so far achieved at {}".format(index))
            best_model_root = os.path.join(config_file.OUTPUT_FOLDER, save_folder, "best_model_folder")
            cmd = "cp -r {}/ {}".format(experiment_folder, best_model_root)
            subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
