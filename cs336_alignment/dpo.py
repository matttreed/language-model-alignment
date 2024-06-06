import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM
import numpy as np
import random
import gzip
import json
from tqdm import tqdm
from cs336_alignment.language_models import get_prompt
from cs336_alignment.sft import get_sft_string


# def get_dpo_string(prompt, response):
#     return (
#         "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
#         "\n"
#         "### Instruction:\n"
#         f"{prompt}\n"
#         "\n"
#         "### Response:\n"
#         f"{response}<|endoftext|>"
#     )

def get_log_prob(logits, labels):
    labels = labels[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

def per_instance_dpo_loss(
        lm: torch.nn.Module | AutoModelForCausalLM,
        lm_ref: torch.nn.Module | AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        beta: float,
        prompt: str,
        response_chosen: str,
        response_rejected: str
    ):

    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    chosen_tokenized = tokenizer.encode(get_sft_string(prompt, response_chosen), add_special_tokens=True, truncation=False, return_tensors="pt")
    rejected_tokenized = tokenizer.encode(get_sft_string(prompt, response_rejected), add_special_tokens=True, truncation=False, return_tensors="pt")

    chosen_log_prob = get_log_prob(lm(chosen_tokenized.to(lm_device)).logits, chosen_tokenized)
    rejected_log_prob = get_log_prob(lm(rejected_tokenized.to(lm_device)).logits, rejected_tokenized)
    ref_chosen_log_prob = get_log_prob(lm_ref(chosen_tokenized.to(ref_device)).logits, chosen_tokenized).to(lm_device)
    ref_rejected_log_prob = get_log_prob(lm_ref(rejected_tokenized.to(ref_device)).logits, rejected_tokenized).to(lm_device)

    pol_relative = chosen_log_prob - rejected_log_prob
    ref_pol_relative = ref_chosen_log_prob - ref_rejected_log_prob

    loss = -F.logsigmoid(beta * (pol_relative.sum() - ref_pol_relative.sum()))

    return loss

# chosen_tokenized = torch.cat((prompt_tokenized, chosen_response_tokenized), dim=-1)
#     rejected_tokenized = torch.cat((prompt_tokenized, rejected_response_tokenized), dim=-1)

# prompt_tokenized = tokenizer.encode(prompt, add_special_tokens=True, truncation=False, return_tensors="pt").to(device)
    # chosen_response_tokenized = tokenizer.encode(response_chosen, add_special_tokens=True, truncation=False, return_tensors="pt").to(device)
    # rejected_response_tokenized = tokenizer.encode(response_rejected, add_special_tokens=True, truncation=False, return_tensors="pt").to(device)

# reward_accuracies = (pref_relative > dispref_relative).float().mean(dim=-1)
#     reward_margins = (pref_relative - dispref_relative).mean(dim=-1)


# print(lm(chosen_tokenized).logits)
    # print(F.softmax(lm(chosen_tokenized).logits, dim=-1))
    # print(chosen_log_prob, ref_chosen_log_prob, rejected_log_prob, ref_rejected_log_prob)
    # print(pref_relative.sum(), dispref_relative.sum())
    # logits = lm(chosen_tokenized).logits[:, :-1, :]
    # torch.gather(F.softmax(logits, dim=-1), dim=2, index=chosen_tokenized[:, 1:]).squeeze(2)


# (pref_relative - dispref_relative) = 0.488

    # def calculate_DPO_loss(model_prefered_logprob, model_disprefered_logprob,
    #                    ref_prefered_logprob, ref_disprefered_logprob,
    #                    beta=0.5):

    # prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    # disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    # reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    # reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    # loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

    # return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

# 0.48

