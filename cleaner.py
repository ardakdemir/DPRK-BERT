import re
import string
from unicodedata import normalize

puncs = string.punctuation
korean_puncs = "〈〉◇】【】《》"
all_punctuations = puncs + korean_puncs

remove_multiple_spaces = lambda sentence: re.sub('(\s)+', " ", sentence).strip()
remove_puncs = lambda sentence: re.sub("[{}]".format(all_punctuations), " ", sentence)

dprk2rok_dict = {"췰": "칠",
                 "돐": "주년",
                 "윁": "베트",
                 "췰": "칠",
                 "꾜": "쿄",
                 "뙈": "떼",
                 "뙤": "뙤",
                 "곬": "골",
                 "¥": "엔",
                 "쩬": "첸",
                 "묜": "먼",
                 "쩰": "첼",
                 "뻰": "펜",
                 "꽛": "단단",
                 "쩬": "첸",
                 "묜": "먼",
                 "쩰": "첼",
                 "럅": "랍",
                 "쇨": "보낼",
                 "섶": "섶",
                 "튐": "탄",
                 "덞": "더러워",
                 "뼁": "펜",
                 "펐": "펐",
                 "틀어쥠으로써": "틀어쥐는 것으로",
                 "섧게도":"서럽게도",
                 "섧은" : "서러운",
                 "돎": "두름",
                 "섧": "서럽",
                 "봏": "볶",
                 "쌴": "산",
                 "뗑": "땡",
                 "볌": "밤",
                 "쨀": "가를",
                 "쇰": "보냄",
                 "쬼": "촘",
                 "겊": "겊",
                 "내밂으로써": "내밀어서",
                 "캤": "캐었",
                 "쥅": "집",
                 "띳":"띳"
                 }

add_to_vocab_list = ["겯", "쯘", "껠", "삯", "쩹", "룁", "긷","겊", "밂", "휑"]


def map_dprk2rok(sentence):
    """
        Mapping DPRK syllables to RoK
    :param sentence:
    :return:
    """
    conversions_sorted = [(n,s) for n,s in dprk2rok_dict.items()]
    conversions_sorted.sort(key=lambda x:len(x[0]),reverse=True)
    for n,s in conversions_sorted:
        sentence = sentence.replace(n,s)
    return sentence

class Cleaner:
    def __init__(self):
        self.name = "Cleaner"

    def normalize(self, sentence):
        return normalize("NFKC", sentence)

    def clean(self, sentence):
        return remove_multiple_spaces(remove_puncs(map_dprk2rok(self.normalize(sentence))))

    def speak(self):
        print("My name is {}.".format(self.name))
