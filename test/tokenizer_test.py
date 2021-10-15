
def tokenizer_test(tokenizer, samples,kwargs={}):
    for sample in samples:
        output = tokenizer.tokenize(sample,kwargs=kwargs)
        print("Input tokens",sample)
        print("Tokenizer Output",output)

def main():
    tokenizer = init_tokenizer()
    kwargs = {}
    samples =["동녘하늘이 희붐히 밝아지더니 불덩이같은","동녘하늘이 희붐히 밝아지더니"]

if __name__ == "__main__":
    main()
