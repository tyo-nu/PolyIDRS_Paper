#!/bin/bash
#SBATCH --account=invpoly
#SBATCH --time=1-00
#SBATCH --job-name=kfold
#SBATCH --nodes=1
#SBATCH -c 18
#SBATCH --gres=gpu:2
#SBATCH --output=./outputs/R10_%j.out  # %j will be replaced with the job ID
#SBATCH --qos=high

source ~/.bashrc
conda activate polyml
module load cudnn/8.1.1/cuda-11.2

for ((i = 0; i < 2 ; i++)); do
    export c=$(($m*2 + $i))
    srun -l -n 1 --gres=gpu:1 --nodes=1 python make_kfold_predictions.py --kfolds "$c" --model_folder "/projects/invpoly/kshebek/stereochemistry_hub/run_results/pm_as_global_mn_exist/220420_af64_bf64_mf8_mg10_DP25_R5_K10" --df_folder "/projects/invpoly/kshebek/stereochemistry_hub/predictions/data_gen/generated_data/220515_PH_copo_R10DP25.csv" --save_folder "./results/220515_PHA_copo_kfold${mf}.csv"
done

wait