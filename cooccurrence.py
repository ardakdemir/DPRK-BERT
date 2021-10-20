import os
import json
from tqdm import tqdm
from collections import defaultdict
from utils import get_documents, save_cooccur_to_txt, save_json, get_exp_name
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Find cooccurrences for a given list of words")
    parser.add_argument(
        "--source_file_path",
        type=str,
        default="/Users/ardaakdemir/dprk_research/dprk-bert-data/new_year_mlm_data/train.json",
        help="The name of the dataset to use for the search.",
    )
    parser.add_argument(
        "--save_file_path",
        type=str,
        default="/Users/ardaakdemir/dprk_research/dprk-bert-data/cooccurrence_dict.json",
        help="Path to save the cooccurrence dictionary",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=0,
        help="Window size to get the neighboring words",
    )
    parser.add_argument(
        "--save_folder_path",
        type=str,
        default="/Users/ardaakdemir/dprk_research/dprk-bert-data/cooccurrence_news",
        help="Path to save the cooccurrence dictionary",
    )
    parser.add_argument("--lemmatize", default=False, action="store_true",
                        help="If set, applies lemmatization to the found words")
    parser.add_argument("--analyze", default=False, action="store_true", help="Find pos tags etc.")
    args = parser.parse_args()
    return args


def cooccurrence(documents, word_list, skip_list, window=0):
    """
            Find all neighbors within window distance for each word in word_list and skip words that contain
            any word from the skip list

            Documents is a list of documents:
                - each document is a collection of sentences separated with newline \n
    :return: dictionary of counts of each cooccurring word for each word in word_list:

            {"Arda": {"Akdemir":100, "javelin":30, "nlp":10},
             "Elon": {"Musk": 250, "SpaceX": 13}}
    """
    cooccurring_words = {}
    for document in tqdm(documents, desc="Document"):
        for sentence in document.split("\n"):
            for word in word_list:
                if word in sentence:
                    sent = sentence.split()
                    for ind, token in enumerate(sent):
                        if word in token and not any([s in token for s in skip_list]):
                            cooccur = [sent[i] for i in
                                       range(max(0, ind - window), min(len(sent), ind + window + 1))]
                            # print("Adding {} as cooccurring words".format(cooccur))
                            if word not in cooccurring_words:
                                cooccurring_words[word] = defaultdict(int)
                            for c in cooccur:
                                cooccurring_words[word][c] += 1
    return cooccurring_words


def main():
    exp_name = get_exp_name()
    args = parse_args()
    p = args.source_file_path
    save_folder_path = args.save_folder_path
    window = args.window_size
    word_list = ["일제","일본","핵","남조선","미제"]
    skip_list = ["일본새"]
    documents = get_documents(p)
    cooccurring_words = cooccurrence(documents, word_list, skip_list, window=window)
    save_cooccur_to_txt(cooccurring_words, save_folder_path)


if __name__ == "__main__":
    main()
