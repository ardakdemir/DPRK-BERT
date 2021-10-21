# DPRK-BERT

An attempt to develop the first open-source DPRK (North Korea) Korean language model.  


Assuming that the files and paths are arranged accordingly.

## Language Model 

***Train.***

For 10 epochs
```
python3 mlm_trainer.py --mode train --num_train_epochs 10 
```

***Evaluate.***

For 5 folds, each fold with 500 batches (per_device_eval_batch_size=8 by default)
```
python3 mlm_trainer.py --mode evaluate --mlm_eval_repeat 5 --validation_steps 500
```


## Text Analysis


***Cooccurrence***

Change the word list and words to skip accordingly in the cooccurrence.py file

    word_list = ["일제","일본","핵","남조선","미제"]
    skip_list = ["일본새"]
    
Then run:

    python cooccurrence.py --source_file_path <PATH-TO-INPUT-FILE> --save_folder_path <PATH-TO-THE-FOLDER-TO-STORE-THE-RESULTS> --window_size <WINDOW_SIZE(0 by default)>
    
Example:     

    python3 cooccurrence.py --source_file_path ../dprk-bert-data/rodong_mlm_training_data/validation.json --save_folder_path ../cooccurrences_rodong_validation 