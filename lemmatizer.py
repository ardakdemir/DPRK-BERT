from soylemma import Lemmatizer

lemmatizer = Lemmatizer()

def test():
    input = '남조선에서'
    func_dict = [("Analyze",lemmatizer.analyze),("Lemmatize",lemmatizer.lemmatize)]
    print("input", input)
    for n,func in func_dict:
        output = func(input)
        print(n,output)