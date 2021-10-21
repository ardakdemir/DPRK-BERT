from soylemma import Lemmatizer
import mecab
mecab = mecab.MeCab()
lemmatizer = Lemmatizer()

def test():
    input = '좋은음식'
    func_dict = [("Analyze",lemmatizer.analyze),("Lemmatize",lemmatizer.lemmatize)]
    print("input", input)
    for n,func in func_dict:
        output = func(input)
        print(n,output)

# test()