from unicodedata import normalize
from tokenizers import Tokenizer as DefTokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFKC, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from transformers import BertTokenizer
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer from a source file")

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="dprk-bert-rodong-1311",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        default="../pretrained_model_data",
        help="Source file for training the tokenizer",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="../pretrained_model_data",
        help="Save folder for dataset",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16424,
        help="vocab size",
    )
    args = parser.parse_args()
    return args


class Tokenizer:
    def __init__(self, tokenizer, cleaner=None, subchar=False):
        self.tokenizer = tokenizer
        self.start_token = "[CLS]"
        self.end_token = "[SEP]"
        self.subchar = subchar
        self.cleaner = cleaner

    def transform(self, string):
        if self.subchar:
            string = normalize("NFKD", string)
        if self.cleaner:
            string = self.cleaner.clean(string)
        tokens = self.tokenizer.tokenize(string)
        tokens = [self.start_token] + tokens + [self.end_token]
        token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        return token_ids, tokens

    def tokenize(self, example, kwargs={}):
        """
        Clean and map DPRK to RoK then tokenize with the KR-BERT tokenizer
        :param examples:
        :param kwargs:
        :return:
        """
        tokenized = []
        if self.cleaner:
            example = self.cleaner.clean(example)
        return self.tokenizer(example, **kwargs)

    def detokenize(self, tokens):
        string = []
        cur_token = ""
        for t in tokens[1:-1]:
            if t.startswith("##"):
                cur_token += t[2:]
            else:
                if len(cur_token) > 0:
                    string.append(cur_token)
                cur_token = t
        return " ".join(string)


def train_bert_tokenizer(training_file_list, vocab_size, tokenizer_save_folder, tokenizer_name):
    bert_tokenizer = DefTokenizer(WordPiece())
    bert_tokenizer.normalizer = normalizers.Sequence([NFKC(), Lowercase(), StripAccents()])
    bert_tokenizer.pre_tokenizer = Whitespace()

    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    trainer = WordPieceTrainer(vocab_size=vocab_size,
                               special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    bert_tokenizer.train(training_file_list, trainer)
    save_path = os.path.join(tokenizer_save_folder, tokenizer_name + ".json")
    model_files = bert_tokenizer.model.save(tokenizer_save_folder, tokenizer_name)
    bert_tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")
    bert_tokenizer.save(save_path)
    return bert_tokenizer


def main():

    args = parse_args()
    source_file = args.source_file
    tokenizer_name = args.tokenizer_name
    save_folder = args.save_folder
    vocab_size = args.vocab_size

    training_file_list = [source_file]
    train_bert_tokenizer(training_file_list, vocab_size, save_folder, tokenizer_name)
    vocab_path = os.path.join(save_folder, tokenizer_name + "-vocab.txt")
    tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
    print("Tokenizer vocab size: {}".format(len(tokenizer.vocab)))


if __name__ == "__main__":
    main()
