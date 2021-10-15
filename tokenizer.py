from unicodedata import normalize

class Tokenizer:
    def __init__(self, tokenizer, cleaner=None,subchar=False):
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

    def tokenize(self,examples,kwargs={}):
        """
            I know it is super weird but works
        :param examples:
        :param kwargs:
        :return:
        """
        tokenized = []
        if self.cleaner:
            examples = list(map(lambda exp : self.cleaner.clean(exp),examples))
        return self.tokenizer(examples[0],**kwargs)

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