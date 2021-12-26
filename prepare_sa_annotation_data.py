"""
    Using keywords generate candidate paragraphs for sentiment annotations...

    What should be our categories?

        Currently: keep it simple...
         - Positive, Negative
        Future:
         - Aggresive, Cooperative, Peaceful, Sympathetic (?)
"""
import os
import json
import uuid
import argparse
import tqdm
import jsonlines

file_map = {"rodong": "/Users/ardaakdemir/dprk_research/dprk-bert-data/rodong_all_data_2210/rodong_all.json",
            "newyear": "/Users/ardaakdemir/dprk_research/dprk-bert-data/new_year_mlm_data/newyear_train.json"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate candidate paragraphs from dprk documents for sa annotation...")
    parser.add_argument(
        "--source_json_file",
        type=str,
        default="../dprk-bert-data/new_year_mlm_data/newyear_train.json",
        help="Source folder containing all data.",
    )
    parser.add_argument(
        "--keyword_file_path",
        type=str,
        default="../dprk-bert-data/sentiment_words.json",
        help="Path to the keyword file.",
    )
    parser.add_argument(
        "--save_type",
        type=str,
        choices=["txt", "json", "jsonl"],
        default="json",
        help="Path to the keyword file.",
    )
    parser.add_argument(
        "--save_file_path",
        type=str,
        default="../dprk-bert-data/paragraphs_for_sentiment_annotation/newyear_sa_positive_paragraphs.txt",
        help="Save txt file.",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="../dprk-bert-data/paragraphs_for_sentiment_annotation_2112",
        help="Save txt file.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="Number of paragraphs to write to the save file",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="(-+) Number of surrounding sentences to get.",
    )
    args = parser.parse_args()
    return args


def write_candidate_sentences(candidates, save_file_path):
    s = []
    for i, c in enumerate(candidates):
        title = "# Paragraph {} id: {} index: {}".format(i, c["id"], c["index"])
        sentiment_words = "Sentiment word(s): {} ".format(
            "\t".join(["{} ({})".format(s["word"], s["value"]) for s in c["sentiment_words"]]))
        s.append("\n".join([title, sentiment_words, c["paragraph"]]))
    with open(save_file_path, "w") as o:
        o.write("\n\n".join(s))


def write_candidate_sentences_json(candidates, save_file_path, save_type="json"):
    data = []
    print(f"saving {len(candidates)} paragraphs to {save_file_path}")
    for i, c in enumerate(candidates):
        my_data = {"text": c["paragraph"],
                   "id": c["id"] + "_" + str(c["index"]),
                   "source":c["source"]}
        data.append(my_data)
    if save_type == "json":
        with open(save_file_path, "w") as o:
            json.dump(data, o)
    if save_type == "jsonl":
        with jsonlines.open(save_file_path, 'w') as writer:
            writer.write_all(data)


def get_candidate_sentences(source_file, keywords, size, window_size, meta=None):
    source_data = json.load(open(source_file))["data"]
    candidates = []
    for d in tqdm.tqdm(source_data, "document"):
        my_id = d.get("id", str(uuid.uuid4())[:8])
        if meta:
            prefix = meta["source"] + "_" + meta["label"]
            my_id = "_".join([prefix, my_id])
        sentences = d["data"].split("\n")
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            if any([k in sent for k, v in keywords.items()]):
                paragraph = "\n".join(sentences[max(0, i - window_size):min(len(sentences), i + window_size + 1)])
                sentiment_words = [v for k, v in keywords.items() if k in paragraph]
                instance_data = {"paragraph": paragraph,
                                 "id": my_id,
                                 "index": i,
                                 "sentiment_words": sentiment_words}
                instance_data.update(meta)
                candidates.append(instance_data)
                i = i + window_size + 1
                if len(candidates) == size: return candidates
            else:
                i = i + 1
    return candidates


def get_keywords(keyword_file_path):
    with open(keyword_file_path) as o:
        return json.load(o)


def main():
    args = parse_args()

    keyword_file_path = args.keyword_file_path
    # source_file = args.source_json_file
    source_file = "../dprk-bert-data/new_year_mlm_data/newyear_train.json"
    save_file_path = args.save_file_path
    size = args.size
    window_size = args.window_size

    save_type = args.save_type
    save_folder = args.save_folder
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    print("Save type", save_type)

    # find candidates + write to file
    # only binary positive or negative for now
    for label in ["pos", "neg"]:
        for k, p in file_map.items():

            save_path = os.path.join(save_folder, "{}_{}_candidates.{}".format(k, label, save_type))
            source_file = p
            keywords = get_keywords(keyword_file_path)
            if label == "pos":
                keywords = {k: v for k, v in keywords.items() if v["value"] > 0}
            elif label == "neg":
                keywords = {k: v for k, v in keywords.items() if v["value"] < 0}
            print("Keywords for finding the candidates", keywords)
            id_prefix = f"{k}_{label}"
            meta = {"source": k, "label": label}
            candidates = get_candidate_sentences(source_file, keywords, size, window_size, meta=meta)
            print(f"Number of candidates for {k} {label} : ", len(candidates))
            if save_type == "txt":
                write_candidate_sentences(candidates, save_path)
            elif save_type in ["json", "jsonl"]:
                write_candidate_sentences_json(candidates, save_path, save_type=save_type)


if __name__ == "__main__":
    main()
