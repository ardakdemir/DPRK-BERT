from unicodedata import normalize
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


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
    bert_tokenizer = Tokenizer(WordPiece())
    bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
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
    model_files = bert_tokenizer.model.save(tokenizer_save_folder, "")
    bert_tokenizer.model = WordPiece.from_file(*model_files, unk_token="[UNK]")
    bert_tokenizer.save(save_path)
    return bert_tokenizer
