#/usr/local/bin/nosh
#$ -cwd
#$ -l v100=1,s_vmem=200G,mem_req=200G
cd ~/dprk-research/DPRK-BERT

experiment_root="../experiment_outputs/"
save_folder=${1}
eval_save_folder=${2}
num_train_epochs=${3}
steps_per_epoch=${4}
validation_steps=${5}

for r_w in 0.5 0.6 0.7 0.8 0.9 1
do
  #Train models
  my_save_folder=${save_folder}"_clregularizer_weight_""${r_w//./-}"
  my_evalsave_folder=${eval_save_folder}"_clregularizer_weight_""${r_w//./-}"
  singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode train --validation_file ../dprk-bert-data/rodong_mlm_training_data/validation.json --validation_file2 ../dprk-bert-data/KoRxnli-test-mlm-ko.json --with_cl_regularization  --regularizer_weight ${r_w}  --num_train_epochs ${num_train_epochs} --validation_steps ${validation_steps}  --steps_per_epoch ${steps_per_epoch} --save_folder ${my_save_folder}

  model_path=${experiment_root}${my_save_folder}"/best_model_weights.pkh"
  #Eval
  singularity exec  --nv  --writable ~/singularity/dprk-image python3 mlm_trainer.py --mode evaluate --save_folder ${my_evalsave_folder} --cross_lingual_model_name_or_path ${model_path} --validation_steps ${validation_steps}
done
