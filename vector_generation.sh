/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
cd ~/dprk-research/DPRK-BERT

experiment_root="../experiment_outputs/"
source_path=${1}
save_folder=${2}
size=${3}

singularity exec  --nv  --writable ~/singularity/dprk-image python3 vector_generation.py --source_json_path ${source_path} --save_folder ${save_folder} --size ${size}
