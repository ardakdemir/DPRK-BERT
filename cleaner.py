import re
import string
from unicodedata import normalize

puncs = string.punctuation
korean_puncs = "〈〉◇】【】《》"
all_punctuations = puncs + korean_puncs

remove_multiple_spaces = lambda sentence: re.sub('(\s)+', " ", sentence).strip()
remove_puncs = lambda  sentence: re.sub("[{}]".format(all_punctuations)," ",sentence)

dprk2rok_dict = {"췰":"칠",
                 "돐":"주년",
                 "윁":"베트",
                 "쯘":"뜬",
                 "췰":"칠"}

def map_dprk2rok(sentence):
    """
        Mapping DPRK syllables to RoK
    :param sentence:
    :return:
    """
    return "".join([dprk2rok_dict.get(x,x) for x in sentence])

class Cleaner:
    def __init__(self):
        self.name = "Cleaner"


    def normalize(self,sentence):
        return normalize("NFKC", sentence)

    def clean(self,sentence):
        return remove_multiple_spaces(remove_puncs(map_dprk2rok(self.normalize(sentence))))

    def speak(self):
        print("My name is {}.".format(self.name))