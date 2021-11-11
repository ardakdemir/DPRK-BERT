import os
import json
import numpy as np


def write_preds_to_txt(path_to_pred_json_file, save_path):
    pred_json = json.load(open(path_to_pred_json_file, "rb"))["0"]
    s = ""
    print("Writing results for {} sentences".format(len(pred_json)))
    for i, sentence in enumerate(pred_json):
        s += "====== Sentence {} ======\n".format(i)
        raw_sentence = sentence["raw_sentence"]
        predictions = sentence["predictions"]
        sentence_with_masks = sentence["sentence_with_masks"]
        s += "Sentence: {}\nSentence with masks: {}\n\n".format(raw_sentence, sentence_with_masks)
        for j, masked_question in enumerate(predictions):
            correct_token = masked_question["masked_word"].replace("#", "")
            s += "## Mask {}\nCorrect token: {}\n".format(j, correct_token)
            for model, value in masked_question["model_data"].items():
                pred = value["top_n_preds"][0].replace("#", "")
                correct_rank = value["correct_rank"]
                correct_prob = round(value["correct_prob"],2)
                s += "{}: {} correct_rank: {} correct_prob: {}\n".format(model, pred, correct_rank, correct_prob)
        s += "\n"
    with open(save_path,"w") as o:
        o.write(s)


path_to_pred_json_file = "/Users/ardaakdemir/dprk_research/dprk-bert-data/error_analysis_results_2510.json"
save_path ="../dprk-bert-data/error-analysis-2610.txt"
pred_json = json.load(open(path_to_pred_json_file, "rb"))
# print(pred_json["0"][0].keys())
sentence = pred_json["0"][0]
prediction_example = sentence["predictions"]
print(sentence["raw_sentence"])
print(sentence["sentence_with_masks"])
print(prediction_example)

write_preds_to_txt(path_to_pred_json_file, save_path)