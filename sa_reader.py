import json
import os
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
from collections import Counter

def read_jsonl(jsonl_file_path):
    with jsonlines.open(jsonl_file_path) as reader:
        return [obj for obj in reader]


def get_labeled_data(jsonlines):
    labeled_data = []
    for l in tqdm(jsonlines, desc="reading data"):
        labels = l["label"]
        id = l["id"]
        source = l["source"]
        text = l['data']
        for label in labels:
            s, e, tag = label
            my_data = {"data": text[s:e],
                       "source": source,
                       "id": id,
                       "tag": tag}
            labeled_data.append(my_data)
    return labeled_data



def read_data_from_folder(folder_path,file_names):
    files = file_names
    print("reading the following files: ",files)
    data_dict = {}
    for f in files:
        jsonl_file_path = os.path.join(folder_path, f + ".jsonl")
        labeled_data = sa_reader(jsonl_file_path)
        data_dict[f] = labeled_data

    for k,v in data_dict.items():
        print(k,len(v))
    return data_dict

def sa_reader(jsonl_file_path):
    data = read_jsonl(jsonl_file_path)
    return data


def write_to_jsonl(data, save_file_path):
    with jsonlines.open(save_file_path, 'w') as writer:
        writer.write_all(data)


def main():
    jsonl_file_path = "/Users/ardaakdemir/dprk_research/dprk-bert-data/sentimentanalysis_dataset_0605.jsonl"
    save_folder = "/Users/ardaakdemir/dprk_research/dprk-bert-data/dprk_sentiment_data_0605"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    split = 0.8
    labels = ["negative-sentence", "positive-sentence"]
    data = read_jsonl(jsonl_file_path)
    labeled_data = get_labeled_data(data)
    save_all_labeled_path = os.path.join(save_folder, "all_labeled.jsonl")
    write_to_jsonl(labeled_data, save_all_labeled_path)
    labeled_data = [l for l in labeled_data if l["tag"] in labels]
    np.random.shuffle(labeled_data)
    labeled_data_grouped = [[x for x in labeled_data if x["tag"]==l] for l in labels]
    print("{} samples for {} tags".format(len(labeled_data), labels))
    train_all,test_all = [],[]
    for lab,grp in zip(labels,labeled_data_grouped):
        cut = int(len(grp) * split)
        train, test = grp[:cut], grp[cut:]
        train_all.extend(train)
        test_all.extend(test)

    for t in [(train_all, "train"), (test_all, "test")]:
        data = t[0]
        np.random.shuffle(data)
        counts = Counter([d["tag"] for d in data])
        print("{} samples for {}".format(len(data), t[1]))
        print(counts)
        p = os.path.join(save_folder, t[1] + ".jsonl")
        write_to_jsonl(data, p)


if __name__ == "__main__":
    main()
