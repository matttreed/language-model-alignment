from cs336_alignment.language_models import MODELPATH_8B, MODELPATH_80B_INSTRUCT, load_model, sample_from_model
from cs336_alignment.mmlu import get_mmlu_prompt, get_mmlu_prompts, parse_mmlu_response
from cs336_alignment.gsm8k import get_gsm8k_prompt, get_gsm8k_prompts, parse_gsm8k_response
from cs336_alignment.alpaca import get_alpaca_prompt, get_alpaca_prompts
from cs336_alignment.sst import get_sst_prompt, get_sst_prompts
import json
import argparse
from copy import copy

def eval_zero_shot_mmlu(model, name):
    print("Loading MMLU Data")
    prompts, data = get_mmlu_prompts()
    print("Data Loaded: " + str(len(prompts)) + " examples")

    outputs = sample_from_model(model=model, prompts=prompts, max_tokens= 256, stop=["# Query:"])

    results = []

    for row, output in zip(data, outputs):
        model_response = output.outputs[0].text
        model_answer = parse_mmlu_response(model_response, options=row["options"])
        results.append({
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "model_answer": model_answer,
            "model_response": model_response,
            "correct": row["answer"] == model_answer
        })  

    correct_answers = sum(result['correct'] for result in results)
    num_no_answers = sum(result["model_answer"] == None for result in results)
    total_questions = len(results)
    accuracy = (correct_answers / total_questions) * 100

    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Questions With No Answer: {num_no_answers}")
    print(f"Accuracy: {accuracy:.2f}%")

    save_path = f"cs336_alignment/scripts/outputs/{name}_mmlu.json"

    with open(save_path, "w") as f:
        json.dump(results, fp=f, indent=4)

    print(f"Saved to: {save_path}")

def eval_zero_shot_gsm8k(model, name):
    print("Loading GSM8K Data")
    prompts, data = get_gsm8k_prompts()
    print("Data Loaded: " + str(len(prompts)) + " examples")

    outputs = sample_from_model(model=model, prompts=prompts, max_tokens=1024, stop=["# Query:"])

    results = []

    for row, output in zip(data, outputs):
        model_response = output.outputs[0].text
        model_answer = parse_gsm8k_response(model_response)
        real_answer = parse_gsm8k_response(row["answer"])
        results.append({
            "question": row["question"],
            "answer": row["answer"],
            "real_answer": real_answer,
            "model_answer": model_answer,
            "model_response": model_response,
            "correct": real_answer == model_answer
        })  

    correct_answers = sum(result['correct'] for result in results)
    num_no_answers = sum(result["model_answer"] == None for result in results)
    total_questions = len(results)
    accuracy = (correct_answers / total_questions) * 100

    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Questions With No Answer: {num_no_answers}")
    print(f"Accuracy: {accuracy:.2f}%")

    save_path = f"cs336_alignment/scripts/outputs/{name}_gsm8k.json"

    with open(save_path, "w") as f:
        json.dump(results, fp=f, indent=4)

    print(f"Saved to: {save_path}")

def eval_zero_shot_alpaca(model, name):
    print("Loading Alpaca Data")
    prompts, data = get_alpaca_prompts()
    print("Data Loaded: " + str(len(prompts)) + " examples")

    outputs = sample_from_model(model=model, prompts=prompts, max_tokens=1024, stop=["# Query:"])

    results = []

    for row, output in zip(data, outputs):
        model_response = output.outputs[0].text
        results.append({
            "instruction": row["instruction"],
            "output": model_response,
            "generator": "llama-3-8b-base",
            "dataset": row["dataset"]
        })

    save_path = f"cs336_alignment/scripts/outputs/{name}_alpaca.json"

    with open(save_path, "w") as f:
        json.dump(results, fp=f, indent=4)

    print(f"Saved to: {save_path}")


def eval_zero_shot_sst(model, name):
    print("Loading SST Data")
    prompts, data = get_sst_prompts()
    print("Data Loaded: " + str(len(prompts)) + " examples")

    outputs = sample_from_model(model=model, prompts=prompts, max_tokens=1024, stop=["# Query:"])

    results = []

    for row, output in zip(data, outputs):
        model_response = output.outputs[0].text
        row_copy = copy(row)
        row_copy["output"] = model_response
        results.append(row_copy)  

    save_path = f"cs336_alignment/scripts/outputs/{name}_sst.jsonl"

    with open(save_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--model-path", type=str, default=MODELPATH_8B, help="path to stored model")
    parser.add_argument("--name", type=str, default="zero_shot", help="name of model")
    parser.add_argument('--mmlu', action="store_true", help='mmlu')
    parser.add_argument('--gsm8k', action="store_true", help='gsm8k')
    parser.add_argument('--alpaca', action="store_true", help='alpaca')
    parser.add_argument('--sst', action="store_true", help='sst')

    args = parser.parse_args()

    print("Loading Model")
    model = load_model(args.model_path)
    print("Model Loaded")

    if args.mmlu:
        eval_zero_shot_mmlu(model, args.name)
    if args.gsm8k:
        eval_zero_shot_gsm8k(model, args.name)
    if args.alpaca:
        eval_zero_shot_alpaca(model, args.name)
    if args.sst:
        eval_zero_shot_sst(model, args.name)