import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
import numpy as np
import random
import gzip
import json
from tqdm import tqdm

END_OF_TEXT_TOKEN = "<|end_of_text|>"

def get_sft_string(prompt, response):
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        "\n"
        "### Instruction\n"
        f"{prompt}\n"
        "\n"
        "### Response\n"
        f"{response}\n"
    )


class SFTDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str, seq_length: int, shuffle: bool =False):
        self.tokenizer = tokenizer 
        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.shuffle = shuffle

        self.tokens = []

        print("Tokenizing Dataset")
        # with gzip.open(dataset_path, 'r') as file:
        with open(dataset_path, 'r') as file:

            lines = file.readlines()
            if shuffle:
                random.shuffle(lines)

            for line in tqdm(lines):
                json_obj = json.loads(line)
                sft_string = get_sft_string(json_obj["prompt"], json_obj["response"]) + END_OF_TEXT_TOKEN
                sft_tokenized = self.tokenizer(sft_string, add_special_tokens=True, truncation=False)['input_ids']
                self.tokens += sft_tokenized

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_length

    def __getitem__(self, i):
        
        start_i = self.seq_length * i
        input_ids = torch.tensor(self.tokens[start_i:start_i+self.seq_length], dtype=torch.long)
        print(len(self.tokens[start_i:start_i+self.seq_length]), i, len(self))
        labels = torch.tensor(self.tokens[start_i + 1:start_i+self.seq_length + 1], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def get_batches(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

