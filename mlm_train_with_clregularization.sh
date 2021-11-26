/usr/local/bin/nosh
#$ -cwd
#$ -l v100=1,s_vmem=200G,mem_req=200G
cd ~/dprk-research/DPRK-BERT

experiment_root="../experiment_outputs/"
save_folder=${1}
eval_save_folder=${2}
num_train_epochs=${3}
#Train models
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode train --with_cl_regularization --num_train_epochs ${num_train_epochs} --save_folder ${save_folder}

model_path=${experiment_root}${save_folder}"/best_model_weights.pkh"
#Eval
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode evaluate --save_folder ${eval_save_folder} --model_name_or_path ${model_path}