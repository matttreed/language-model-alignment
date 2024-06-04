from cs336_alignment.language_models import get_prompt
import pandas as pd
import os
import json
import re

GSM8K_TEST_PATH = "data/gsm8k/test.jsonl"

def load_gsm8k_dataset(path=GSM8K_TEST_PATH):
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(
                {
                    "question": json_obj["question"],
                    "answer": json_obj["answer"]
                }
            )
    return data

def get_gsm8k_prompt(question):
    gsm8k_instruction = (
        f"{question}\n"
        f"Answer:\n"
    )

    return get_prompt(instruction=gsm8k_instruction)

def get_gsm8k_prompts():
    data = load_gsm8k_dataset()
    questions = [get_gsm8k_prompt(ex["question"]) for ex in data]
    return questions, data

def parse_gsm8k_response(response: str):
    numbers = re.findall(r'\d+', response)
    if numbers:
        return int(numbers[-1])
    else:
        return None