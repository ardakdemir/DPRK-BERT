import os
import json


def cooccurrence(word_list,skip_list,window=0):
    """
        Find all words in the window neighborhood of all words in the word_list excluding the skip_list words
        window 0 means in the same token (without blank). E.g,
                - word:조선반도에서 평화를 보장하고 민족자주
    :param word_list:
    :param skip_list:
    :param window:
    :return:
    """