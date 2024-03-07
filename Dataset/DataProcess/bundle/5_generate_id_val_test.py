import json
import pandas as pd
import os
import argparse
import re
import numpy as np

def extract_item(string, size, use_test=False):
    matches = re.findall(r'\d+\."([^"]+)"', string)
    if use_test:
        return matches
    else:
        return matches[:-size]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate id')
    parser.add_argument('--seed', 
                        type=int,
                        default=42,
                        help='options: 42, 625, 2023, 0, 10')
    parser.add_argument('--candidate_size', 
                        type=int,
                        default=20,
                        help='the size of candidate size')
    
    args = parser.parse_args()

    save_path = './final_dataset/LLM/total/'
    with open(f'{save_path}train_seed_{args.seed}_size_{args.candidate_size}.json', 'r') as json_file:
        train = json.load(json_file)
    with open(f'{save_path}val_seed_{args.seed}_size_{args.candidate_size}.json', 'r') as json_file:
        val = json.load(json_file)
    with open(f'{save_path}test_seed_{args.seed}_size_{args.candidate_size}.json', 'r') as json_file:
        test = json.load(json_file)
    
    item_map = np.load('./final_dataset/item_map_bundle.npy', allow_pickle=True).item()
    
    total_data = [val, test]
    candidate_id = []
    data_session = []
    save_path = './final_dataset/ID/'
    for idx, data in enumerate(total_data):
        for sess_idx, lst in enumerate(data):
            item_list = extract_item(lst['input'], args.candidate_size, use_test=True)
            session_list = item_list[:-args.candidate_size]
            candidate_list = item_list[-args.candidate_size:]
            tmp = []
            for item in candidate_list:
                tmp.append(item_map[item])
            candidate_id.append(tmp)
            session_list.append(lst['target'])
            tmp = []
            for item in session_list:
                tmp.append(item_map[item])
            data_session.append(tmp)
        if idx == 0:
            np.save(f"{save_path}valid_seed_{args.seed}_itemnum_{len(item_map)}_bundle.npy", data_session)
            np.save(f'./final_dataset/ID/valid_candidate_{args.seed}_bundle.npy', candidate_id)
        else:
            np.save(f"{save_path}test_seed_{args.seed}_itemnum_{len(item_map)}_bundle.npy", data_session)
            np.save(f'./final_dataset/ID/test_candidate_{args.seed}_bundle.npy', candidate_id)
        
        


