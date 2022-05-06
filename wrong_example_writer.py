import json
import os
from sa_trainer import write_wrong_examples, model_name_map, init_tokenizer
from tokenizer import Tokenizer

model_map = {"mbert": {"tokenizer": model_name_map["mbert"],
                       "from_pretrained": True,
                       "file_path": "/Users/ardaakdemir/dprk_research/experiment_outputs/sentiment_analysis_1001_1255_mbert/train_summary.json"},
             "kr-bert": {
                 "tokenizer": None,
                 "from_pretrained": False,
                 "file_path": "/Users/ardaakdemir/dprk_research/experiment_outputs/sentiment_analysis_1001_1255_krbert/train_summary.json"},
             "dprk-bert": {
                 "tokenizer": None,
                 "from_pretrained": False,
                 "file_path": "/Users/ardaakdemir/dprk_research/experiment_outputs/sentiment_analysis_1001_1255_dprkbert/train_summary.json"}
             }


def write_single(save_folder, model_name):
    print("Writing results for ", model_name)
    save_path = os.path.join(save_folder, "wrong_examples_{}.txt".format(model_name))
    model_info = model_map[model_name]
    my_tokenizer, name = init_tokenizer(model_info["tokenizer"], model_info["from_pretrained"])
    with open(model_info["file_path"]) as r:
        res = json.load(r)["best_result"]
    wrong_examples = res["wrong_examples"]
    write_wrong_examples(wrong_examples, my_tokenizer, save_path)

save_folder = "../wrong_examples_1301"
if not os.path.exists(save_folder):os.makedirs(save_folder)

for k in model_map.keys():
    model_name = k
    write_single(save_folder, model_name)
