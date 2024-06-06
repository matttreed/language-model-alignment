#!/bin/bash                                                                                                                                                                    
#SBATCH --job-name=vscode
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --time=2:00:00
#SBATCH --output=cs336_alignment/scripts/logs/sst_eval_%j.out
#SBATCH --error=cs336_alignment/scripts/logs/sst_eval_%j.err
#SBATCH --mem=100G  

python scripts/evaluate_safety.py \
--input-path cs336_alignment/scripts/outputs/sft_sst.jsonl \
--model-name-or-path /home/shared/Meta-Llama-3-70B-Instruct \
--num-gpus 2 \
--output-path cs336_alignment/scripts/outputs/sst_sft_eval_outputs.jsonl