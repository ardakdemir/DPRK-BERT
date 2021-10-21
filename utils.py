import json
import os
import config
import datetime as dt
from cleaner import Cleaner

def get_documents(document_json_path):
    cleaner = Cleaner()
    with open(document_json_path, "r") as r:
        r = json.load(r)
        return [cleaner.clean(d["data"]) for d in r["data"]]


def remove_stopwords(documents, stopwords, skip_list):
    for s in skip_list:
        substrings = get_substrings(s)
        for sub in substrings:
            if sub in stopwords:
                stopwords.remove(sub)
    for i, d in enumerate(documents):
        for s in stopwords:
            d = d.replace(" "+s+" ", "")
        documents[i] = d
    return documents


def get_substrings(string):
    return ["".join(string[s:e]) for s in range(len(string)) for e in range(s + 1, len(string))]


def save_json(dict, save_path):
    with open(save_path, "w") as o:
        json.dump(dict, o)


def save_cooccur_to_txt(cooccur_dict, save_folder):
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    for c, counts in cooccur_dict.items():
        counts = [(w, f) for w, f in counts.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        s = "Word\tCount\n" + "\n".join(["{}\t{}".format(w, f) for w, f in counts])
        save_path = os.path.join(save_folder, c + ".txt")
        with open(save_path, "w") as o:
            o.write(s)


def get_exp_name(prefix=None):
    day, time = dt.datetime.now().isoformat().split("T")
    suffix = "_".join([day, "-".join(time.split(".")[0].split(":"))])
    if prefix is None:
        return suffix
    else:
        return "_".join([prefix, suffix])
