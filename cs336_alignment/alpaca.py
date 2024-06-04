from cs336_alignment.language_models import get_prompt
import pandas as pd
import os
import json

ALPACA_TEST_PATH = "data/alpaca_eval/alpaca_eval.jsonl"

def load_alpaca_dataset(path=ALPACA_TEST_PATH):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(
                {
                    "instruction": json_obj["instruction"],
                    "output": json_obj["output"],
                    "dataset": json_obj["dataset"]
                }
            )
    return data

def get_alpaca_prompt(instruction):
    return get_prompt(instruction=instruction)

def get_alpaca_prompts():
    data = load_alpaca_dataset()
    questions = [get_alpaca_prompt(ex["instruction"]) for ex in data]
    return questions, data