from soylemma import Lemmatizer

import time
from konlpy.tag import Kkma
from konlpy.utils import pprint
from collections import Counter


lemmatizer = Lemmatizer()
kkma = Kkma()


def get_nouns(sentences):
    return [kkma.nouns(s) for s in sentences]


def test():
    input = '일제의'
    func_dict = [("Analyze", lemmatizer.analyze), ("Lemmatize", lemmatizer.lemmatize)]
    print("input", input)
    for n, func in func_dict:
        output = func(input)
        print(n, output)


def test_kkma():
    """
        input
            - 시대착오적 적대시 정책  English: anachronistic hostile policy
    :return:
    """
    # sentence = u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'
    sentence = "조선민주주의인민공화국에 대한 시대착오적 적대시 정책 "
    # sentence = "터키 공화국" #republic of Turkey
    pprint(kkma.pos(sentence, flatten=False))


def main():
    # test_kkma()
    sentences = ["고립되어 가고 있습니"]
    nouns = [x for y in get_nouns(sentences) for x in y]
    noun_counts = Counter(nouns)
    print(noun_counts)
if __name__ == "__main__":
    main()
