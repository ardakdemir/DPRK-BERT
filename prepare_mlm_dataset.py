import os
import json
from tqdm import tqdm
import argparse
import config

"""
Dataset is a list of dictionary entries each entry having a "data" field that contains the main text and
any other fields optionally.

E.g.,

dataset = {"data":[
                    {"data":"Main text of my document","title":"Arda","date":"12-08-2021","id":"my_id"},
                    {"data":"Main text of my document2","title":"Akdemir","date":"13-08-2021","id":"my_id2"}
                  ]
          } 
"""

data_store_field = "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate json datasets for mlm training/testing ")
    parser.add_argument(
        "--input_type",
        type=str,
        default="rodong",
        choices=["rodong", "new_year"],
        help="Type of data for preparation",
    )
    parser.add_argument(
        "--apply_split",
        action="store_true",
        default=False,
        help="Whether to split the dataset into train/validation",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Split ratio when split is true",
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        default="../rodong_parser/data/parsed_rodong_pages_1210_1923",
        help="Source folder containing all data",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="rodong_mlm_training_data_1912",
        help="Save folder for dataset",
    )
    args = parser.parse_args()
    return args


def prepare_newyear_data(source_folder, save_folder, split=0.8):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    all_year_data_names = [x for x in os.listdir(source_folder) if x.endswith("txt")]
    data = []
    for file_name in tqdm(all_year_data_names, desc="years"):
        p = os.path.join(source_folder, file_name)
        year = file_name.split(".")[0]
        with open(p, "r") as r:
            d = r.read()
        my_data = {"id": year,
                   "data": d,
                   "date": year,
                   "year":int(year),
                   "type": "newyear_speech",
                   "source": "newyear"}
        data.append(my_data)
    t = int(len(data) * split)
    tuples = [('train.json', data[:t]), ('validation.json', data[t:])]
    for tup in tuples:
        if len(tup[1]) > 0:
            with open(os.path.join(save_folder, tup[0]), "w") as o:
                json.dump({"data": tup[1]}, o)
        else:
            print('no data for {} split'.format(tup[0].split(".")[0]))


def get_all_day_data(source_folder, save_folder, split=0.8):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    all_day_data_paths = [os.path.join(source_folder, x) for x in os.listdir(source_folder) if x.endswith("json")]
    data = []
    for p in tqdm(all_day_data_paths, desc="days"):
        with open(p, "rb") as r:
            d = json.load(r)
            values = list(d.values())
            new_values = []
            for v in values:
                my_v = {"id": v["news_id"],
                        "date": v["date"],
                        "year":int(v["date"].split("-")[0]),
                        "title": v["title"],
                        "source": "rodong",
                        "type": v["type"]}
                if v["type"] == "news_article":
                    my_v["data"] = v["article"]
                else:
                    my_v["data"] = v["letter"]
                new_values.append(my_v)
            data.extend(new_values)
    t = int(len(data) * split)
    tuples = [('train.json', data[:t]), ('validation.json', data[t:])]
    for tup in tuples:
        if len(tup[1]) > 0:
            with open(os.path.join(save_folder, tup[0]), "w") as o:
                json.dump({"data": tup[1]}, o)
        else:
            print('no data for {} split'.format(tup[0].split(".")[0]))


def main():
    args = parse_args()
    input_type = args.input_type

    save_folder = args.save_folder
    source_folder = args.source_folder
    split_ratio = args.split_ratio
    apply_split = args.apply_split
    if input_type == "rodong":
        get_all_day_data(source_folder, save_folder, split=split_ratio if apply_split else 1)
    elif input_type == "new_year":
        prepare_newyear_data(source_folder, save_folder, split=split_ratio if apply_split else 1)


if __name__ == "__main__":
    main()
