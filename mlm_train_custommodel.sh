/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/dprk-research/DPRK-BERT

experiment_root="../experiment_outputs/"
save_folder=${1}
eval_save_folder=${2}
num_train_epochs=${3}
vocab_path=${4}
config_path=${5}
train_file=${6}

#Train models
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode train --num_train_epochs ${num_train_epochs} --save_folder ${save_folder} --tokenizer_name ${vocab_path} --config_name ${config_path} --train_file ${train_file} --train_from_scratch

model_path=${experiment_root}${save_folder}"/best_model_weights.pkh"
#Eval
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode evaluate --save_folder ${eval_save_folder} --model_name_or_path ${model_path} --tokenizer_name ${vocab_path} --config_name ${config_path}