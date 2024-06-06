#!/bin/bash                                                                                                                                                                    
#SBATCH --job-name=vscode
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --time=2:00:00
#SBATCH --output=cs336_alignment/scripts/logs/alpaca_eval_%j.out
#SBATCH --error=cs336_alignment/scripts/logs/alpaca_eval_%j.err
#SBATCH --mem=100G  

alpaca_eval --model_outputs cs336_alignment/scripts/outputs/sft_alpaca.json \
--annotators_config 'scripts/alpaca_eval_vllm_llama3_70b_fn' \
--base-dir '.'