import json
import os
import config
import datetime as dt

def get_documents(document_json_path):
    with open(document_json_path,"r") as r:
        r = json.load(r)
        return [d["data"] for d in r["data"]]

def save_json(dict,save_path):
    with open(save_path,"w") as o:
        json.dump(dict,o)

def save_cooccur_to_txt(cooccur_dict, save_folder):
    if not os.path.exists(save_folder):os.makedirs(save_folder)
    for c , counts in cooccur_dict.items():
        counts = [(w,f) for w,f in counts.items()]
        counts.sort(key = lambda x : x[1], reverse=True)
        s = "Word\tCount\n"+ "\n".join(["{}\t{}".format(w,f) for w,f in counts])
        save_path = os.path.join(save_folder,c+".txt")
        with open(save_path,"w") as o:
            o.write(s)

def get_exp_name(prefix=None):
    day, time = dt.datetime.now().isoformat().split("T")
    suffix = "_".join([day, "-".join(time.split(".")[0].split(":"))])
    if prefix is None:
        return suffix
    else:
        return "_".join([prefix, suffix])