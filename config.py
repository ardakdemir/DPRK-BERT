import os

DATA_ROOT = "../dprk-bert-data"

RODONG_MLM_ROOT = os.path.join(DATA_ROOT,"rodong_mlm_training_data")
LOG_FILE_PATH = os.path.join(DATA_ROOT,"log_file_mlmtraining.log")


VOCAB_FILE_PATH = "../kr-bert-pretrained/vocab_snu_char16424.pkl"
WEIGHT_FILE_PATH = "../kr-bert-pretrained/pytorch_model_char16424_bert.bin"
CONFIG_FILE_PATH = "../kr-bert-pretrained/bert_config_char16424.json"
VOCAB_PATH = "../kr-bert-pretrained/vocab_snu_char16424.txt"

mlm_train_json = os.path.join(RODONG_MLM_ROOT,"train.json")
mlm_validation_json = os.path.join(RODONG_MLM_ROOT,"validation.json")

STOPWORDS_PATH = "misc/korean_stopwords.txt"

OUTPUT_FOLDER = "../experiment_outputs"


