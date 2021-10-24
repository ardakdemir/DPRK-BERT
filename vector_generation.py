import pickle as pkl
import numpy as np
from document import Document, Sentence
import torch
from utils import get_document_objects, save_objects_to_pickle, load_pickle
from mlm_trainer import init_mlm_models_from_dict
from tqdm import tqdm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "--source_json_path",
        type=str,
        default="../dprk-bert-data/new_year_mlm_data/train.json",
        help="Source folder containing all data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="../dprk-bert-data/newyear_document_vectors.pkl",
        help="Save folder for dataset",
    )
    args = parser.parse_args()
    return args


def generate_vectors(document_objects, model_dicts, device):
    """
        For each sentence generate a feature vector using all models
    :param sentence_objects:
    :param model_dicts:
    :return:
    """
    for k, model_dict in model_dicts.items():
        model_dict["model"].to(device)
        model_dict["model"].eval()
    progress_bar = tqdm(range(len(document_objects) * len(model_dicts)), desc="Step")
    for document in document_objects:
        document_sentences = document.get_sentences()
        document_id = document.get_id()
        num_sents = len(document_sentences)
        print("{} sentences for {}".format(num_sents, document_id))
        for sentence in document_sentences:
            raw_sentence = sentence.raw_sentence
            for k, model_dict in model_dicts.items():
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
    return document_objects


def generate_vectors_test(documents_json_path, save_path):
    dprk_model_path = "../experiment_outputs/2021-10-17_02-36-11/best_model_weights.pkh"
    model_dict = {"KR-BERT": {
        "model_name": "../kr-bert-pretrained/pytorch_model_char16424_bert.bin",
        "tokenizer": None,
        "config_name": None},
        "DPRK-BERT": {"model_name": dprk_model_path,
                      "tokenizer": None, "config_name": None},
        "KR-BERT-MEDIUM": {"model_name": "snunlp/KR-Medium",
                           "tokenizer": "snunlp/KR-Medium",
                           "config_name": "snunlp/KR-Medium",
                           "from_pretrained": True},
        "mBERT": {"model_name": "bert-base-multilingual-cased",
                  "tokenizer": "bert-base-multilingual-cased",
                  "config_name": "bert-base-multilingual-cased",
                  "from_pretrained": True}
    }
    print("initializing the models...")
    model_dicts = init_mlm_models_from_dict(model_dict)
    document_objects = get_document_objects(documents_json_path)
    document_objects = generate_vectors(document_objects, model_dicts, device)

    print("Saving sentence vectors to ", save_path)
    save_objects_to_pickle(document_objects, save_path)

    documents = load_pickle(save_path)
    sentence = documents[0].sentences[0]
    print("{} documents {} sentences".format(len(documents), len(documents[0].sentences)))
    print("Sentence id", sentence.sentence_id)
    print("Sentence metadata", sentence.metadata)
    print(len(sentence.vectors))


def main():
    args = parse_args()
    source_json_path = args.source_json_path
    save_path = args.save_path
    generate_vectors_test(source_json_path, save_path)


if __name__ == "__main__":
    main()
