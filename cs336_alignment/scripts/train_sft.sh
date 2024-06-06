#!/bin/bash                                                                                                                                                                    
#SBATCH --job-name=vscode
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=cs336_alignment/scripts/logs/train_%j.out
#SBATCH --error=cs336_alignment/scripts/logs/train_%j.err
#SBATCH --mem=100G       

export WANDB_API_KEY="a881776c91971a761af061cba1423c1d6e38ab60"

eval "$(conda shell.bash hook)"
conda activate cs336_alignment

# torchrun --standalone --nproc_per_node=2 /home/c-mattreed/language-model-alignment/cs336_alignment/scripts/train_sft.py \
# --train-path /home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz \
# --dev-path /home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz \
# --output-dir /home/c-mattreed/language-model-alignment/models \
# --batch_size 2 \
# --gradient-accumulation-steps 16 \
# --sequence-length 512 \
# --eval-iters 1000 \
# --eval-interval 1000 \
# --learning-rate 1e-3 \
# --lr-scheduler cosine \
# --weight-decay 0.1 \
# --warmup-ratio 0.01 \
# --grad-clip 1.0 \
# --dtype bfloat16 \
# --wandb-project cs336-sft \
# --compile \
# --device cuda

# python /home/c-mattreed/language-model-alignment/cs336_alignment/scripts/train_sft.py \
# --train-path /home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz \
# --dev-path /home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz \
# --output-dir /home/c-mattreed/language-model-alignment/models \
# --batch-size 2 \
# --gradient-accumulation-steps 16 \
# --sequence-length 512 \
# --eval-iters 1000 \
# --eval-interval 1000 \
# --learning-rate 2e-5 \
# --lr-scheduler cosine \
# --weight-decay 0.1 \
# --warmup-ratio 0.01 \
# --grad-clip 1.0 \
# --dtype bfloat16 \
# --wandb-project cs336-sft \
# --device cuda \
# --model-name-or-path "/data/Meta-Llama-3-8B"

python /home/c-mattreed/language-model-alignment/cs336_alignment/scripts/train_sft.py \
--train-path /home/shared/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz \
--dev-path /home/shared/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz \
--output-dir /home/c-mattreed/language-model-alignment/models_2 \
--batch-size 2 \
--gradient-accumulation-steps 16 \
--sequence-length 512 \
--eval-iters 1000 \
--eval-interval 1000 \
--learning-rate 2e-5 \
--lr-scheduler cosine \
--weight-decay 0.1 \
--warmup-ratio 0.01 \
--grad-clip 1.0 \
--dtype bfloat16 \
--wandb-project cs336-sft \
--device cuda \
--model-name-or-path "/data/Meta-Llama-3-8B" \
--epochs 2 \
--shuffle-data