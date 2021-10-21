from soylemma import Lemmatizer

import time
from konlpy.tag import Kkma
from konlpy.utils import pprint

lemmatizer = Lemmatizer()
kkma = Kkma()


def test():
    input = '일제의'
    func_dict = [("Analyze",lemmatizer.analyze),("Lemmatize",lemmatizer.lemmatize)]
    print("input", input)
    for n,func in func_dict:
        output = func(input)
        print(n,output)

def test_kkma():
    """
        input
            - 시대착오적 적대시 정책  English: anachronistic hostile policy
    :return:
    """
    # sentence = u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'
    sentence = "조선민주주의인민공화국에 대한 시대착오적 적대시 정책 "
    sentence = "터키 공화국" #republic of Turkey
    pprint(kkma.pos(sentence,flatten=False))

# test()
test_kkma()