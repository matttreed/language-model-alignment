import torch
from torch.nn.functional import sigmoid
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM
import numpy as np
import random
import gzip
import json
from tqdm import tqdm


def per_instance_dpo_loss(
        lm: torch.nn.Module | AutoModelForCausalLM,
        lm_ref: torch.nn.Module | AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        beta: float,
        prompt: float,
        response_chosen: str,
        response_rejected: str,
    ):
    chosen_tokenized = torch.tensor(tokenizer(prompt + response_chosen, add_special_tokens=True, truncation=False)["input_ids"], dtype=torch.long)
    rejected_tokenized = torch.tensor(tokenizer(prompt + response_rejected, add_special_tokens=True, truncation=False)["input_ids"], dtype=torch.long)
    chosen_logits = lm(chosen_tokenized).logits
    rejected_logits = lm(rejected_tokenized).logits
    ref_chosen_logits = lm_ref(chosen_tokenized).logits
    ref_rejected_logits = lm_ref(rejected_tokenized).logits

    winning = chosen_logits - ref_chosen_logits
    losing = rejected_logits - ref_rejected_logits

    loss = - torch.log(sigmoid(beta * (torch.mean(winning) - torch.mean(losing))))

    return loss

