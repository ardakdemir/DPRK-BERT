#/usr/local/bin/nosh
#$ -cwd
#$ -l v100=1,s_vmem=200G,mem_req=200G
cd ~/dprk-research/DPRK-BERT

#root_folder_prefix: mlmtrain_hypersearch_clregular_2711_1144_train_clregularizer_weight_
experiment_root="../experiment_outputs/"
root_folder_prefix=${1}
#eval_save_folder=${2}
#num_train_epochs=${3}
#steps_per_epoch=${4}

for r_w in "0-4" "0-5" "0-6" "0-7" "0-8" "0-9" "1"
do
  #Train models
  echo ${r_w}
#  model_path=${experiment_root}${my_save_folder}"/best_model_weights.pkh"
  python3 mlm_trainer.py --mode evaluate --mlm_eval_repeat 3 --validation_steps 500 --validation_file ../dprk-bert-data/KoRxnli-test-mlm-ko.json --save_folder kornli-mlmtest_2711_2141_rw_${r_w} --cross_lingual_model_name_or_path ../experiment_outputs/${root_folder_prefix}0-1/best_model_weights.pkh
done
