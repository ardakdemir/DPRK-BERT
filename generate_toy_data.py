import os
import json


def generate_toy_data(source, destination, size=10):
   with open(source,"rb") as r:
       d = json.load(r)

       document = d["data"][0]
       save_dict = {k: v for k, v in document.items()}
       text = "\n".join(document["data"].split("\n")[:size])
       save_dict["data"] = text
       save_dict = {"data":[save_dict]}
       with open(destination,"w") as o:
           json.dump(save_dict,o)


def main():
    source = "/Users/ardaakdemir/dprk_research/dprk-bert-data/rodong_all_data_2210/train.json"
    destination = "/Users/ardaakdemir/dprk_research/dprk-bert-data/rodong_all_data_2210/train_toy.json"

    generate_toy_data(source, destination, size=10)

if __name__ == '__main__':
    main()