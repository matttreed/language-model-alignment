from cs336_alignment.language_models import MODELPATH_7B, MODELPATH_80B_INSTRUCT, load_model, sample_from_model
from cs336_alignment.mmlu import get_mmlu_prompt, get_mmlu_prompts, parse_mmlu_response
import json

def main():
    print("Loading Data")
    prompts, data = get_mmlu_prompts()
    print("Data Loaded: " + str(len(prompts)) + " examples")

    print("Loading Model")
    model = load_model(MODELPATH_7B)
    print("Model Loaded")

    outputs = sample_from_model(model=model, prompts=prompts, max_tokens= 256, stop=["# Query:"])

    results = []

    for row, output in zip(data, outputs):
        model_response = output.outputs[0].text
        model_answer = parse_mmlu_response(model_response)
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

    with open("cs336_alignment/scripts/outputs/zero_shot_mmlu.json", "w") as f:
        json.dump(results, fp=f, indent=4)




if __name__ == "__main__":
    main()