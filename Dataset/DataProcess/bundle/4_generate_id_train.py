import json
import argparse
import re
import os
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
    
    item_map = np.load('./final_dataset/item_map_bundle.npy', allow_pickle=True).item()
    
    
    candidate_id = []
    save_path = './final_dataset/ID/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_session = []
    for sess_idx, lst in enumerate(train):
        session_list = extract_item(lst['input'], args.candidate_size, use_test=False)
        session_list.append(lst['target'])
        tmp = []
        for item in session_list:
            tmp.append(item_map[item])
        data_session.append(tmp)
    data_session = np.array(data_session)
    np.save(f"{save_path}train_itemnum_{len(item_map)}_bundle.npy", data_session)


