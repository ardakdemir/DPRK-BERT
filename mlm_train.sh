/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/dprk-research/DPRK-BERT

experiment_root="../experiment_outputs/"
save_folder=${1}
eval_save_folder=${2}
num_train_epochs=${3}
steps_per_epoch=${4}


#Train models
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode train --num_train_epochs ${num_train_epochs} --steps_per_epoch ${steps_per_epoch} --save_folder ${save_folder}

model_path=${experiment_root}${save_folder}"/best_model_weights.pkh"

#Eval Rodong
my_evalsave_folder=${eval_save_folder}"_rodong"
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode evaluate --save_folder ${my_evalsave_folder} --model_name_or_path ${model_path}

#Eval kornli
my_evalsave_folder=${eval_save_folder}"_kornli"
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode evaluate --validation_file ../dprk-bert-data/KoRxnli-test-mlm-ko.json --save_folder ${my_evalsave_folder} --model_name_or_path ${model_path}

#Eval newyear
my_evalsave_folder=${eval_save_folder}"_newyear"
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode evaluate --validation_file ../dprk-bert-data/new_year_mlm_data/newyear_train.json --save_folder ${my_evalsave_folder} --model_name_or_path ${model_path}
