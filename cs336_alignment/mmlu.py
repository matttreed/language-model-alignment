from cs336_alignment.language_models import get_prompt
import pandas as pd
import os

MMLU_TEST_DIR_PATH = "data/mmlu/test"

def load_mmlu_dataset(dir=MMLU_TEST_DIR_PATH):
    file_paths = [os.path.join(dir, filename) for filename in os.listdir(dir) if filename.endswith('.csv')]

    data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        subject = os.path.basename(file_path).split('.')[0].replace("_", " ")
        for row in df.itertuples():
            data.append(
                {
                    "subject": subject,
                    "question": row[1],
                    "options": row[2:6],
                    "answer": row[6]
                }
            )
    
    return data

def get_mmlu_prompt(subject, question, options):
    mmlu_instruction = (
        f"Answer the following multiple choice question about {subject}. "
        "Respond with a single sentence of the form \"The correct answer is _\", filling the blank with the letter "
        "corresponding to the correct answer (i.e., A, B, C or D).\n\n"
        f"Question: {question}\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        f"Answer:\n"
    )

    return get_prompt(instruction=mmlu_instruction)

def get_mmlu_prompts():
    data = load_mmlu_dataset()
    questions = [get_mmlu_prompt(ex["subject"], ex["question"], ex["options"]) for ex in data]
    return questions, data

def parse_mmlu_response(response: str):
    index = response.find("The correct answer is ")
    
    if index != -1:
        answer = response[index + 22]
        
        if answer in 'ABCD':
            return answer
    
    return None