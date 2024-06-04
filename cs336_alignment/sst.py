from cs336_alignment.language_models import get_prompt
import pandas as pd
import os

SST_TEST_PATH = "data/simple_safety_tests/simple_safety_tests.csv"

def load_sst_dataset(path=SST_TEST_PATH):

    data = []

    df = pd.read_csv(path)
    for row in df.itertuples():
        num, sst_id,harm_area,counter,category,prompts_final = row
        data.append(
            {
                "sst_id": sst_id,
                "harm_area": harm_area,
                "counter": counter,
                "category": category,
                "prompts_final": prompts_final
            }
        )
    
    return data

def get_sst_prompt(sst_instruction):
    return get_prompt(instruction=sst_instruction)

def get_sst_prompts():
    data = load_sst_dataset()
    questions = [get_sst_prompt(ex["prompts_final"]) for ex in data]
    return questions, data

# def parse_sst_response(response: str, options=None):
#     index = response.find("The correct answer is ")
    
#     if index != -1:
#         answer = response[index + 22]
        
#         if answer in 'ABCD':
#             return answer

#     # for option, letter in zip(options, ["A", "B", "C", "D"]):
#     #     if option in response:
#     #         return letter
    
#     return None