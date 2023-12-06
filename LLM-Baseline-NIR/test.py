import time
import json
import openai
import random 
import argparse
import re
import os
from metric import Metric
from tqdm import tqdm


def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def openai_requeset(content, openai_model):
    message = [{"role": "user", "content": content}]
    response = openai.ChatCompletion.create(
                    model=openai_model,
                    messages = message,
                    temperature=0.2,
                    frequency_penalty=0.0,
                    )
    return response

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help='0/10/42/625/2023')
parser.add_argument('--dataset', type=str, default="bundle", help="bundle/games/ml-1m")
parser.add_argument('--api_key', type=str, default="", help="openai token")
args = parser.parse_args()

rseed = args.seed
random.seed(rseed)

with open(f"../Dataset/{args.dataset}/Text/test_seed_{args.seed}.json", 'r') as json_file:
    test_data = json.load(json_file)
openai.api_key = args.api_key
openai_model = 'gpt-3.5-turbo'

temp_1 = """
Candidate Set (candidate items): {}.
The items I have interacted with: {}.
Step 1: What features are most important to me when selecting items (Summarize my preferences briefly)? 
Answer: 
"""

temp_2 = """
Candidate Set (candidate items): {}.
The items I have interacted with: {}.
Step 1: What features are most important to me when selecting items (Summarize my preferences briefly)? 
Answer: {}.
Step 2: Selecting the most featured items from the interacted items according to my preferences (Format: [no. an interacted item.]). 
Answer: 
"""


temp_3 = """
CCandidate Set (candidate items): {}.
The items I have interacted with: {}.
Step 1: What features are most important to me when selecting items (Summarize my preferences briefly)? 
Answer: {}.
Step 2: Selecting the most featured items from the interacted items according to my preferences (Format: [no. an interacted item.]). 
Answer: {}.
Step 3: Can you rerank the 20 items from the Candidate Set based on their similarity to the selected items I've interacted with (Format: [no. an interacted item - a candidate item])?.
Answer: 
"""

count = 0
total = 0
results_data = []
for data in tqdm(test_data):
    
    candidate_items = re.findall(r'\d+\."([^"]+)"', data['input'].split('\nCandidate set: ')[1])
    history_items = re.findall(r'\d+\."([^"]+)"', data['input'].split('\nCandidate set: ')[0].split('Current session interactions: ')[1])
    input_1 = temp_1.format(', '.join(candidate_items), ', '.join(history_items))

    try_nums = 5
    kk_flag = 1
    while try_nums:
        try:
            response = openai_requeset(input_1, openai_model)
            try_nums = 0
            kk_flag = 1
        except Exception as e:
            if 'exceeded your current quota' in str(e):
                openai.api_key = args.api_key
            time.sleep(1) 
            try_nums = try_nums-1
            kk_flag = 0

    if kk_flag == 0:
        time.sleep(5) 
        response = openai_requeset(input_1, openai_model)

    predictions_1 = response["choices"][0]["message"]["content"]
    
    
    input_2 = temp_2.format(', '.join(candidate_items), ', '.join(history_items), predictions_1)

    try_nums = 5
    kk_flag = 1
    while try_nums:
        try:
            response = openai_requeset(input_2, openai_model)
            try_nums = 0
            kk_flag = 1
        except Exception as e:
            if 'exceeded your current quota' in str(e):

                # open_ai_keys_index +=1
                openai.api_key = args.api_key
            time.sleep(1) 
            try_nums = try_nums-1
            kk_flag = 0

    if kk_flag == 0:
        time.sleep(5) 
        response = openai_requeset(input_2, openai_model)

    predictions_2 = response["choices"][0]["message"]["content"]
    
    
    input_3 = temp_3.format(', '.join(candidate_items), ', '.join(history_items), predictions_1, predictions_2)

    try_nums = 5
    kk_flag = 1
    while try_nums:
        try:
            response = openai_requeset(input_3, openai_model)
            try_nums = 0
            kk_flag = 1
        except Exception as e:
            if 'exceeded your current quota' in str(e):

                # open_ai_keys_index +=1
                openai.api_key = args.api_key
            time.sleep(1) 
            try_nums = try_nums-1
            kk_flag = 0

    if kk_flag == 0:
        time.sleep(5) 
        response = openai_requeset(input_3, openai_model)

    predictions = response["choices"][0]["message"]["content"]
    

    hit_=0
    if data['target'] in predictions:
        count += 1
        hit_ = 1
    else:
        pass
    total +=1
    

    result_json = {"Input_1": input_1,
                   "Input_2": input_2,
                   "Input_3": input_3,
                   "GT": data['target'],
                   "Predictions_1": predictions_1,
                   "Predictions_2": predictions_2,
                   "Predictions": predictions,
                   'Hit': hit_}
    results_data.append(result_json)

res_dir = f'./res/{args.dataset}/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
file_dir = f"{res_dir}/seed_{args.seed}.json"
write_json(results_data, file_dir)

rank_list = []
for data in results_data:
    hit = data['Hit']
    if hit == 0:
        rank_list.append(21)
    else:
        target = data['GT']
        res_string = data['Predictions']
        res_list = res_string.split('\n')
        for idx, item in enumerate(res_list):
            if target in item:
                if idx >= 20:
                    rank_list.append(21)
                else:
                    rank_list.append(idx + 1)
                break

metric = Metric(rank_list)
res = metric.run()
res.to_csv(f"./res/{args.dataset}/seed_{args.seed}_metric.csv", index=False)



