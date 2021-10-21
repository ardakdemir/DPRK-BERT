/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/dprk-research/DPRK-BERT

num_train_epochs=20
save_folder=${1}
eval_save_folder=${2}
#Train models
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode train --num_train_epochs 1 --save_folder ${save_folder}

model_path=${save_folder}"/best_model_weights.pkh"
#Eval
singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode evaluate --save_folder ${eval_save_folder} --model_name_or_path ${model_path}