import pickle as pkl
import numpy as np
from document import Document, Sentence
import torch
from utils import get_document_objects, save_documents_to_pickle, save_objects_to_pickle, load_pickle, \
    get_google_translations
from mlm_trainer import init_mlm_models_from_dict
from tqdm import tqdm
import argparse
from collections import defaultdict
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate json datasets for mlm training/testing ")
    parser.add_argument(
        "--action",
        type=str,
        default="store_documents",
        choices=["store_documents", "store_nouns"],
        help="Type of data for preparation",
    )
    parser.add_argument(
        "--source_json_path",
        type=str,
        default="../dprk-bert-data/rodong_mlm_training_data/train_toy.json",
        help="Source folder containing all data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../dprk-bert-data/rodong_toy_document_vectors.pkl",
        help="Save folder for dataset",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="../dprk-bert-data/rodong_toy_pickles",
        help="Save folder for dataset",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1e9,
        help="Number of documents to parse",
    )
    args = parser.parse_args()
    return args


def generate_sentence_vectors(document_objects, save_folder, model_dicts, device, number_of_documents=1e9):
    """
        For each sentence generate a feature vector using all models
    :param sentence_objects:
    :param model_dicts:
    :return:
    """
    for k, model_dict in model_dicts.items():
        model_dict["model"].to(device)
        model_dict["model"].eval()
    document_objects.sort(key=lambda x: x.metadata["date"])
    progress_bar = tqdm(range(min(len(document_objects), number_of_documents) * len(model_dicts)), desc="Step")
    documents_to_pickle = []
    pickle_size = 50
    for i, document in enumerate(document_objects):
        if i >= number_of_documents:
            break
        if i + 1 % pickle_size == 0:
            if len(documents_to_pickle) > 0:
                save_documents_to_pickle(documents_to_pickle,save_folder)
                documents_to_pickle = []
        document_sentences = document.get_sentences()
        document_id = document.get_id()
        num_sents = len(document_sentences)
        print("{} sentences for {}".format(num_sents, document_id))
        for sentence in document_sentences:
            raw_sentence = sentence.raw_sentence
            for k, model_dict in model_dicts.items():
                max_seq_length = getattr(model_dict["config"], "max_position_embeddings", 512)
                batch = model_dict["tokenizer"].tokenize(raw_sentence, {"truncation": True,
                                                                        "max_length": max_seq_length})
                model = model_dict["model"]
                with torch.no_grad():
                    batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()}
                    outputs = model(**batch, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    v = hidden_states[-1][0][
                        0].detach().cpu().numpy()  # last layer's first sentence's first token output
                    sentence.set_vector(v, k)
            progress_bar.update(1)
        documents_to_pickle.append(document)
    if len(documents_to_pickle) > 0:
        save_documents_to_pickle(documents_to_pickle, save_folder)
        documents_to_pickle = []
    return document_objects


def generate_word_vectors(word_list, model_dicts, device):
    """
        For each sentence generate a feature vector using all models
    :param sentence_objects:
    :param model_dicts:
    :return:
    """
    for k, model_dict in model_dicts.items():
        model_dict["model"].to(device)
        model_dict["model"].eval()
    progress_bar = tqdm(range(len(word_list) * len(model_dicts)), desc="Step")
    word_vectors = defaultdict(dict)
    for word in word_list:
        for k, model_dict in model_dicts.items():
            max_seq_length = getattr(model_dict["config"], "max_position_embeddings", 512)
            batch = model_dict["tokenizer"].tokenize(word, {"truncation": True,
                                                            "max_length": max_seq_length})
            model = model_dict["model"]
            with torch.no_grad():
                batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()}
                outputs = model(**batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                v = hidden_states[-1][0][
                    0].detach().cpu().numpy()  # last layer's first sentence's first token output
                word_vectors[word][k] = v
            progress_bar.update(1)
    return word_vectors


def generate_noun_vectors_caller(source_json_path, save_path):
    dprk_model_path = "../experiment_outputs/2021-10-17_02-36-11/best_model_weights.pkh"
    # model_dict = {"KR-BERT": {
    #     "model_name": "../kr-bert-pretrained/pytorch_model_char16424_bert.bin",
    #     "tokenizer": None,
    #     "config_name": None},
    #     "DPRK-BERT": {"model_name": dprk_model_path,
    #                   "tokenizer": None, "config_name": None},
    #     "KR-BERT-MEDIUM": {"model_name": "snunlp/KR-Medium",
    #                        "tokenizer": "snunlp/KR-Medium",
    #                        "config_name": "snunlp/KR-Medium",
    #                        "from_pretrained": True},
    #     "mBERT": {"model_name": "bert-base-multilingual-cased",
    #               "tokenizer": "bert-base-multilingual-cased",
    #               "config_name": "bert-base-multilingual-cased",
    #               "from_pretrained": True}
    # }
    model_dict = {
        "DPRK-BERT": {"model_name": dprk_model_path,
                      "tokenizer": None, "config_name": None}
    }
    print("initializing the models...")
    model_dicts = init_mlm_models_from_dict(model_dict)
    document_objects = get_document_objects(source_json_path)
    all_nouns = list(set([n for d in document_objects for n in d.nouns]))
    print("{} documents and {} nouns.".format(len(document_objects), len(all_nouns)))
    word_vectors = generate_word_vectors(all_nouns, model_dicts, device)

    # Get English translations
    print("Getting the translations...")
    trans_begin = time.time()
    translations = get_google_translations(all_nouns)
    trans_end = time.time()
    trans_time = round(trans_end - trans_begin, 3)
    print("Translation time: {} secs".format(trans_time))

    word_dict = {}
    for i, n in enumerate(all_nouns):
        word_dict[n] = {"vectors": word_vectors[n],
                        "translation": translations[i]}
    print("Saving word dict to ", save_path)
    save_objects_to_pickle(word_dict, save_path)

    word_vectors = load_pickle(save_path)
    words = list(word_vectors.keys())
    print("{} words".format(len(word_vectors)))
    print("Value for {}: {}".format(words[0], word_vectors[words[0]]))


def generate_sentence_vectors_test(documents_json_path, save_folder, args):
    size = args.size
    dprk_model_path = "../experiment_outputs/2021-10-17_02-36-11/best_model_weights.pkh"
    # model_dict = {"KR-BERT": {
    #     "model_name": "../kr-bert-pretrained/pytorch_model_char16424_bert.bin",
    #     "tokenizer": None,
    #     "config_name": None},
    #     "DPRK-BERT": {"model_name": dprk_model_path,
    #                   "tokenizer": None, "config_name": None},
    #     "KR-BERT-MEDIUM": {"model_name": "snunlp/KR-Medium",
    #                        "tokenizer": "snunlp/KR-Medium",
    #                        "config_name": "snunlp/KR-Medium",
    #                        "from_pretrained": True},
    #     "mBERT": {"model_name": "bert-base-multilingual-cased",
    #               "tokenizer": "bert-base-multilingual-cased",
    #               "config_name": "bert-base-multilingual-cased",
    #               "from_pretrained": True}
    # }
    model_dict = {
        "DPRK-BERT": {"model_name": dprk_model_path,
                      "tokenizer": None, "config_name": None},
    }
    print("initializing the models...")
    model_dicts = init_mlm_models_from_dict(model_dict)
    document_objects = get_document_objects(documents_json_path)
    document_objects = generate_sentence_vectors(document_objects, save_folder, model_dicts, device,
                                                 number_of_documents=size)

    # print("Saving sentence vectors to ", save_path)
    # save_objects_to_pickle(document_objects, save_path)
    #
    # documents = load_pickle(save_path)
    # sentence = documents[0].sentences[0]
    # print("{} documents {} sentences".format(len(documents), len(documents[0].sentences)))
    # print("Sentence id", sentence.sentence_id)
    # print("Sentence metadata", sentence.metadata)
    # print(len(sentence.vectors))


def main():
    args = parse_args()
    source_json_path = args.source_json_path
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    action = args.action
    if action == "store_documents":
        generate_sentence_vectors_test(source_json_path, save_folder, args)
    elif action == "store_nouns":
        generate_noun_vectors_caller(source_json_path, save_path)


if __name__ == "__main__":
    main()
