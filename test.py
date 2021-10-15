from test.tokenizer_test import tokenizer_test
from tokenizer import Tokenizer
from config import VOCAB_PATH, CONFIG_FILE_PATH, WEIGHT_FILE_PATH
from mlm_trainer import init_tokenizer

def test_tokenizer():
    tokenizer = init_tokenizer()
    kwargs = {}
    samples = ["동녘하늘이 희붐히 밝아지더니 불덩이같은", "동녘하늘이 희붐히 밝아지더니"]
    tokenizer_test(tokenizer,samples,kwargs)


def main():
    test_tokenizer()
if __name__ == "__main__":
    main()