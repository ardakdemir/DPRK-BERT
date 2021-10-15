import os
import json
from tqdm import tqdm
save_folder = "rodong_mlm_training_data"
source_folder = "parsed_rodong_pages_1210_1923"

data_store_field = "data"

def get_all_day_data(source_folder,save_folder,split=0.8):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    all_day_data_paths = [os.path.join(source_folder,x) for x in os.listdir(source_folder) if x.endswith("json")]
    data = []
    for p in tqdm(all_day_data_paths,desc="days"):
        with open(p,"rb") as r:
            d = json.load(r)
            values = list(d.values())
            new_values = []
            for v in values:
                if v["type"] == "news_article":
                    v["data"] = v["article"]
                    my_v = {"text":v["article"],"date":v["date"],"title":v["title"]}
                else:
                    my_v = {"text": v["letter"], "date": v["date"], "title": v["title"]}
                new_values.append(my_v)
            data.extend(new_values)
    t = int(len(data)*0.8)
    tuples = [('train.json',data[:t]),('validation.json',data[t:])]
    for tup in tuples:
        with open(os.path.join(save_folder,tup[0]),"w") as o:
            json.dump({"data":tup[1]},o)

def main():
    save_folder = "rodong_mlm_training_data"
    source_folder = "parsed_rodong_pages_1210_1923"
    get_all_day_data(source_folder, save_folder, split=0.8)
if __name__ == "__main__":
    main()