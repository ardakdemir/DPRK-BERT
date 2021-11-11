import json
import os
import config
import datetime as dt
from cleaner import Cleaner
from document import Document
import pickle
from googletrans import Translator

translator = Translator(service_urls=[
    'translate.google.co.kr'
])


def get_google_translations(sentences):
    return [translator.translate(s, src="ko").text for s in sentences]


def get_documents(document_json_path):
    cleaner = Cleaner()
    with open(document_json_path, "r") as r:
        r = json.load(r)
        return [cleaner.clean(d["data"]) for d in r["data"]]


def get_document_objects(documents_json_path):
    cleaner = Cleaner()
    document_objects = []
    with open(documents_json_path, "r") as r:
        documents = json.load(r)
        for d in documents["data"]:
            data = d["data"]
            sentences = data.split("\n")
            kwargs = {k: v for k, v in d.items() if k != "data"}
            kwargs["document_id"] = kwargs["id"]
            kwargs["document_type"] = kwargs["type"]
            # print("{} sentences for {}".format(len(sentences),kwargs["document_id"]))
            doc = Document(sentences, kwargs)
            document_objects.append(doc)
    return document_objects


def remove_stopwords(documents, stopwords, skip_list):
    for s in skip_list:
        substrings = get_substrings(s)
        for sub in substrings:
            if sub in stopwords:
                stopwords.remove(sub)
    for i, d in enumerate(documents):
        for s in stopwords:
            d = d.replace(" " + s + " ", "")
        documents[i] = d
    return documents


def get_substrings(string):
    return ["".join(string[s:e]) for s in range(len(string)) for e in range(s + 1, len(string))]


def save_json(dict, save_path):
    with open(save_path, "w") as o:
        json.dump(dict, o)


def save_objects_to_pickle(objects, save_path):
    root = os.path.split(save_path)[0]
    if not os.path.exists(root): os.makedirs(root)
    with open(save_path, "wb") as f:
        pickle.dump(objects, f)


def load_pickle(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)


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


def document_object_test():
    documents_json_path = "../dprk-bert-data/new_year_mlm_data/newyear_train.json"
    documents_json_path = "../dprk-bert-data/rodong_mlm_training_data/train_toy.json"
    documents = get_document_objects(documents_json_path)
    first_document = documents[0]
    print(first_document, first_document.get_id(), first_document.get_year(), first_document.get_type())

    first_sentence = first_document.sentences[0]
    print(first_sentence, first_sentence.sentence_id, first_sentence.document_id)


def main():
    document_object_test()


if __name__ == "__main__":
    main()
