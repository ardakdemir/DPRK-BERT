from soylemma import Lemmatizer

lemmatizer = Lemmatizer()

def test():
    input = '좋은음식'
    func_dict = [("Analyze",lemmatizer.analyze),("Lemmatize",lemmatizer.lemmatize)]
    print("input", input)
    for n,func in func_dict:
        output = func(input)
        print(n,output)

test()