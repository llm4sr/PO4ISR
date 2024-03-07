import numpy as np
import pandas as pd
import random
import csv
import json
import re
import gzip
from datetime import datetime
from tqdm import tqdm

if __name__ == '__main__':
    seed = 42 #0, 10, 42. 625, 2023
    sample_num = 50
    random.seed(seed)

    item_map = np.load('./final_dataset/item_map_bundle.npy', allow_pickle=True).item()

    with open(f"./final_dataset/LLM/total/train_seed_{seed}_size_20.json", 'r') as json_file:
        train_data = json.load(json_file)

    train_sample = random.sample(train_data, sample_num)

    with open(f"./final_dataset/LLM/total/train_{sample_num}.json", 'w') as json_file:
        json.dump(train_sample, json_file)
    
    train_sample_id = []
    for data in train_sample:
        session_string = data['input'].split('\nCandidate set')[0]
        items = re.findall(r'\d+\."([^"]+)"', session_string)
        ids = [item_map[item] for item in items]
        train_sample_id.append(ids)
    np.save(f"./final_dataset/ID/train_itemnum_{len(item_map)}_sample_{sample_num}_bundle.npy", train_sample_id)

    with open(f"./final_dataset/LLM/total/valid_seed_{seed}_size_20.json", 'r') as json_file:
        valid_data = json.load(json_file)

    sample_num = 100
    valid_sample = random.sample(valid_data, sample_num)

    with open(f"./final_dataset/LLM/total/valid_{sample_num}.json", 'w') as json_file:
        json.dump(valid_sample, json_file)
    
    valid_sample_id = []
    candidates = []
    for data in valid_sample:
        target = data['target']
        session_string = data['input'].split('\nCandidate set')[0]
        items = re.findall(r'\d+\."([^"]+)"', session_string)
        ids = [item_map[item] for item in items] + [item_map[target]]
        valid_sample_id.append(ids)
        session_string = data['input'].split('\nCandidate set')[1]
        items = re.findall(r'\d+\."([^"]+)"', session_string)
        ids = [item_map[item] for item in items]
        candidates.append(ids)
    
    np.save(f"./final_dataset/ID/valid_itemnum_{len(item_map)}_sample_{sample_num}_bundle.npy", valid_sample_id)
    np.save(f"./final_dataset/ID/valid_sample_{sample_num}_candidate_bundle.npy", candidates)