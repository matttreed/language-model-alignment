#!/usr/bin/env python3
from __future__ import annotations
"""
Train a language model on one or multiple GPUs.

To run single-GPU training:

```
python scripts/train.py
```

To run multi-GPU training, use `torchrun`. e.g., for single-node, 2 GPU:

```
torchrun --standalone --nproc_per_node=2 scripts/train.py
```
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
import logging
import os
import pathlib
import sys
from contextlib import nullcontext

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from cs336_alignment.sft import get_batches, SFTDataset
from cs336_alignment.util import get_cosine_lr


logger = logging.getLogger(__name__)


def train(
    train_path,
    dev_path,
    output_dir,
    sequence_length,
    batch_size,
    gradient_accumulation_steps,
    eval_iters,
    eval_interval,
    learning_rate,
    lr_scheduler,
    warmup_ratio,
    weight_decay,
    adam_beta1,
    adam_beta2,
    adam_eps,
    grad_clip,
    device,
    compile,
    dtype,
    wandb_project,
    shuffle_data,
    model_name_or_path
):
    logger.info("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    logger.info("Tokenizing Datasets")
    train_data = SFTDataset(tokenizer, train_path, sequence_length, shuffle_data)
    dev_data = SFTDataset(tokenizer, dev_path, sequence_length, shuffle_data)

    seed = 0

    logger.info(
        "Total number of tokens per training step: "
        + str(
            gradient_accumulation_steps
            * batch_size
            * sequence_length
        )
    )

    # Seed each process differently so we can be sure that they
    # see different data batches.
    # NOTE: This assumes that you're using torch RNG, you may have
    # to seed numpy too as well if your code uses numpy random functions.
    torch.manual_seed(seed)

    device_type = "cuda" if "cuda" in device else "cpu"
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    logger.info(f"Using dtype: {torch_dtype}")

    logger.info("Loading Model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).to(device_type)
    logger.info("Model loaded")
    # GradScaler is only used for FP16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Move model to the device
    model = model.to(device)

    # compile the model, requires torch 2.0
    if compile:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    # Set up the AdamW optimizer.
    # We do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )

    train_steps = len(train_data) // batch_size

    for i, train_batch in enumerate(tqdm(get_batches(train_data, batch_size, shuffle_data))):
        if lr_scheduler.lower() == "cosine":
            lr = get_cosine_lr(
                i,
                max_learning_rate=learning_rate,
                min_learning_rate=learning_rate * 0.1,
                warmup_iters=int(train_steps * warmup_ratio),
                cosine_cycle_iters=train_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = learning_rate

        # with amp_ctx:
        input_ids = train_batch["input_ids"].to(device)
        labels = train_batch["labels"].to(device)
        logits = model(input_ids).logits
        loss = (
            F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            / gradient_accumulation_steps
        )
        scaler.scale(loss).backward()

        if (i + 1) % gradient_accumulation_steps == 0:

            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss_float = loss.item() * gradient_accumulation_steps
            logger.info(f"Train step {i}, Loss: {loss_float}")
            if wandb_project:
                wandb.log({"train_loss": loss_float, "lr": lr}, step=i)

        if i != 0 and i % eval_interval == 0:
            dev_loss = estimate_dev_loss(
                model=model,
                dev_dataset=dev_data,
                batch_size=batch_size,
                eval_iters=eval_iters,
                device=device,
            )
            logger.info(f"Estimated validation loss: {dev_loss}")
            if wandb_project:
                wandb.log({"eval_loss": dev_loss}, step=i)

    dev_loss = estimate_dev_loss(
        model=model,
        dev_dataset=dev_data,
        batch_size=batch_size,
        eval_iters=eval_iters,
        device=device,
    )
    logger.info(f"Final estimated validation loss: {dev_loss}")
    if wandb_project:
        wandb.log({"eval_loss": dev_loss}, step=train_steps)

    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)

    



@torch.no_grad()
def estimate_dev_loss(
    model: AutoModelForCausalLM,
    dev_dataset: SFTDataset,
    batch_size: int,
    eval_iters: int,
    device: str,
):
    model.eval()
    eval_iters = min(eval_iters, len(dev_dataset))
    losses = torch.zeros(eval_iters)

    for i, batch in enumerate(get_batches(dev_dataset, batch_size, True)):
        if i >= eval_iters:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids).logits  # Forward pass
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses[i] = loss.item()

    model.train()
    return losses.mean()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="Path to input IDs to train with.",
    )
    parser.add_argument(
        "--dev-path",
        required=True,
        help="Path to input IDs to use for measuring validation performance.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to folder to write model configuration and trained model checkpoint",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        required=True,
        help="Sequence length to use when training language model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help=("Batch size to use during training."),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=1,
        type=int,
        help=(
            "Number of forward+backward passes to do with given "
            "batch size for each single train step"
        ),
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=200,
        help="Number of evaluation batches to use for calculating validation loss",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Measure validation loss every `eval-interval` trainig steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help=("Learning rate to use during training."),
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["constant", "cosine"],
        default="cosine",
        help=("Learning rate scheduler to use during training."),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.01,
        help=("Ratio of total steps to use for LR warmup"),
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-1, help="AdamW weight decay"
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
        help=("Value to use for Adam beta_1"),
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.98,
        help=("Value to use for Adam beta_2"),
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-9,
        help=("Value to use for Adam epsilon"),
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        help=("If set, clip gradient norms to this value"),
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0', etc.)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If true, compile the model with torch.compile",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
        help="dtype to use when training",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="If set, log results to the specified wandb project",
    )
    parser.add_argument(
        "--shuffle-data",
        action="store_true",
        help="Shuffle Datasets"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        help="Name or path of model to be finetuned",
    )
    args = parser.parse_args()


    logger.info("running %s", " ".join(sys.argv))

    # Make the directory for output if it doesn't already exist
    if os.path.exists(os.path.join(args.output_dir, "model.pt")):
        raise ValueError(
            f"output directory {args.output_dir} already exists and contains model.pt"
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb_project:
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project=args.wandb_project,
            config=vars(args),
            name=pathlib.Path(args.output_dir).name,
        )

    train(
        args.train_path,
        args.dev_path,
        args.output_dir,
        args.sequence_length,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.eval_iters,
        args.eval_interval,
        args.learning_rate,
        args.lr_scheduler,
        args.warmup_ratio,
        args.weight_decay,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
        args.grad_clip,
        args.device,
        args.compile,
        args.dtype,
        args.wandb_project,
        args.shuffle_data,
        args.model_name_or_path
    )
    logger.info("finished running %s", sys.argv[0])
