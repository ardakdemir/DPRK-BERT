import os

import numpy as np
from lemmatizer import get_nouns
from cleaner import Cleaner

cleaner = Cleaner()

class Sentence:
    def __init__(self, raw_sentence, kwargs={}):
        self.raw_sentence = raw_sentence
        self.nouns = get_nouns([cleaner.clean(raw_sentence)])[0]
        self.vectors = {}  # {"model_name": <vector>}
        try:
            self.document_id = kwargs["document_id"]
            self.sentence_id = kwargs["sentence_id"]
        except Exception as e:
            raise Exception(
                "Required fields (document_id,sentence_id) are not given as kwargs when creating a sentence object")
        self.metadata = kwargs

    def set_vector(self, v, model_name):
        self.vectors[model_name] = v

    def get_vector(self, model_name):
        return self.vectors[model_name]

    def get_metadata(self):
        return self.metadata

    def get_nouns(self):
        return self.nouns


class Document:
    def __init__(self, sentence_list, kwargs):
        self.sentence_list = sentence_list
        self.required_fields = ["document_id", "document_type", "year"]
        try:
            for f in self.required_fields:
                assert f in kwargs
            self.metadata = kwargs
        except Exception as e:
            msg = ",".join(self.required_fields)
            raise Exception(
                f"Required fields ({msg}) are not given as kwargs when creating a sentence object")
        self.init_sentences(self.sentence_list)
        self.init_nouns()

    def get_sentence(self, index):
        return self.sentences[index]

    def get_year(self):
        return self.metadata["year"]

    def get_id(self):
        return self.metadata["document_id"]

    def get_type(self):
        return self.metadata["document_type"]

    def get_sentences(self):
        return self.sentences

    def init_sentences(self, sentence_list):
        sentences = []
        for i, s in enumerate(sentence_list):
            year = self.get_year()
            document_id = self.get_id()
            document_type = self.get_type()
            sentence_id = document_id + "_" + str(i)

            sentence = Sentence(s,
                                {"sentence_id": sentence_id,
                                 "document_id": document_id,
                                 "year": year,
                                 "document_type": document_type})
            sentences.append(sentence)
        self.sentences = sentences

    def init_nouns(self):
        nouns = set()
        for s in self.sentences:
            for n in s.nouns:
                nouns.add(n)
        self.nouns = nouns
