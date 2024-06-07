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
from torch.utils.data import Dataset, DataLoader
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        response_rejected: str,
        max_length: int | None = None
        # logger=None
    ):

    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    chosen_tokenized = tokenizer.encode(get_sft_string(prompt, response_chosen) + tokenizer.eos_token, add_special_tokens=True, truncation=False, return_tensors="pt", max_length=max_length)
    rejected_tokenized = tokenizer.encode(get_sft_string(prompt, response_rejected) + tokenizer.eos_token, add_special_tokens=True, truncation=False, return_tensors="pt", max_length=max_length)

    chosen_log_prob = get_log_prob(lm(chosen_tokenized.to(lm_device)).logits, chosen_tokenized.to(lm_device))
    rejected_log_prob = get_log_prob(lm(rejected_tokenized.to(lm_device)).logits, rejected_tokenized.to(lm_device))

    with torch.no_grad():
        ref_chosen_log_prob = get_log_prob(lm_ref(chosen_tokenized.to(ref_device)).logits, chosen_tokenized.to(ref_device)).to(lm_device)
        ref_rejected_log_prob = get_log_prob(lm_ref(rejected_tokenized.to(ref_device)).logits, rejected_tokenized.to(ref_device)).to(lm_device)

    chosen_prob_diff = chosen_log_prob - ref_chosen_log_prob
    rejected_prob_diff = rejected_log_prob - ref_rejected_log_prob

    loss = -F.logsigmoid(beta * (chosen_prob_diff.sum() - rejected_prob_diff.sum()))

    reward_accuracies = (chosen_prob_diff.mean(dim=-1) > rejected_prob_diff.mean(dim=-1)).float().mean(dim=-1)

    # if logger:
    #     logger.info("YOYOYOYO")
    #     logger.info(loss)
    #     logger.info(chosen_log_prob)
    #     logger.info(rejected_log_prob)
    #     logger.info(chosen_prob_diff)
    #     logger.info(rejected_prob_diff)

    return loss, reward_accuracies



def extract_first_conversation(text):
        human_pattern = re.search(r'Human:(.*?)(?=Assistant:|$)', text, re.DOTALL)
        if human_pattern:
            human_text = human_pattern.group(1).strip()
        else:
            human_text = None

        assistant_pattern = re.search(r'Assistant:(.*?)(?=Human:|$)', text, re.DOTALL)
        if assistant_pattern:
            assistant_text = assistant_pattern.group(1).strip()
        else:
            assistant_text = None

        return human_text, assistant_text

def get_train_eval_sets_dpo(dataset_dir: str, tokenizer: AutoTokenizer,shuffle=True):

    filenames = os.listdir(dataset_dir)

    sets = []

    for filename in filenames:
        with gzip.open(os.path.join(dataset_dir, filename), 'r') as file:
            for line in file:
                json_obj = json.loads(line)
                chosen = json_obj["chosen"]
                rejected = json_obj["rejected"]

                if (
                    chosen.count("Human:") != 1 or chosen.count("Assistant:") != 1
                    or rejected.count("Human:") != 1 or rejected.count("Assistant:") != 1
                ):
                    continue

                prompt, chosen_response = extract_first_conversation(chosen)
                prompt_2, rejected_response = extract_first_conversation(rejected)

                sets.append((prompt, chosen_response, rejected_response))

    if shuffle:
        random.shuffle(sets)

    train_set = DPODataset(sets=sets[200:])
    eval_set = DPODataset(sets=sets[:200])

    return train_set, eval_set

class DPODataset(Dataset):
    def __init__(self, sets):

        self.sets = sets # (chosen, rejected)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, i):
        prompt, chosen_tensor, rejected_tensor = self.sets[i]
        return prompt, chosen_tensor, rejected_tensor
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def get_batches(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("models")
    train_set, eval_set = get_train_eval_sets_dpo("/home/shared/hh", tokenizer, False)

    print(len(train_set))
    print(len(eval_set))

    a, b, c = ["a", "1", "whoa"]

    for batch in get_batches(train_set, 2, shuffle=False):
        prompts, chosen_responses, rejected_responses = batch
        print(prompts)
        print(chosen_responses)
        print(rejected_responses)
        input()

# chosen_tokenized = torch.cat((prompt_tokenized, chosen_response_tokenized), dim=-1)
#     rejected_tokenized = torch.cat((prompt_tokenized, rejected_response_tokenized), dim=-1)

# prompt_tokenized = tokenizer.encode(prompt, add_special_tokens=True, truncation=False, return_tensors="pt").to(device)
    # chosen_response_tokenized = tokenizer.encode(response_chosen, add_special_tokens=True, truncation=False, return_tensors="pt").to(device)
    # rejected_response_tokenized = tokenizer.encode(response_rejected, add_special_tokens=True, truncation=False, return_tensors="pt").to(device)

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

