#!/bin/bash                                                                                                                                                                    
#SBATCH --job-name=vscode
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --time=2:00:00
#SBATCH --output=cs336_alignment/scripts/logs/sft_eval_%j.out
#SBATCH --error=cs336_alignment/scripts/logs/sft_eval_%j.err
#SBATCH --mem=100G       

eval "$(conda shell.bash hook)"
conda activate cs336_alignment

python /home/c-mattreed/language-model-alignment/cs336_alignment/scripts/eval_model.py \
--model-path /home/c-mattreed/language-model-alignment/models \
--mmlu \
--gsm8k \
--alpaca \
--sst \
--name sft