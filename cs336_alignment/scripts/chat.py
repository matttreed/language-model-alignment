#!/usr/bin/env python3

from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.language_models import get_prompt

import argparse

import torch


def chat(
    model_name_or_path,
    device_type,
    max_tokens,
    num_responses,
    temperature
):
    print("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print("Loading Model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
    ).to(device_type)
    print("Model loaded")

    while True:
        text = input("Prompt for model (enter to exit):\n")
        if not text:
            return
        
        text = get_prompt(text)
        
        # tokenized = torch.tensor(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"], dtype=torch.long).to(device_type)
        tokenized = tokenizer.encode(text, return_tensors="pt").to(device_type)

        output = model.generate(tokenized, max_length=max_tokens, num_return_sequences=num_responses, temperature=temperature)

        generated_texts = tokenizer.batch_decode(output, skip_special_tokens=False)
        for i, text in enumerate(generated_texts):
            print(f"Generated text {i+1}: {text}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="models",
        help="Path to model",
    )
    parser.add_argument(
        "--device-type",
        default="cuda",
        help="Path to input IDs to train with.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens generated",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=1,
        help="Max responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature for generation",
    )
    args = parser.parse_args()

    chat(
        args.model_path,
        args.device_type,
        args.max_tokens,
        args.num_responses,
        args.temperature
    )
