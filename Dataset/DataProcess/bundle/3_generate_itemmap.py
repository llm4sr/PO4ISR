import json
import re
import numpy as np
import argparse

def extract_item(string, size, use_test=False):
    matches = re.findall(r'\d+\."([^"]+)"', string)
    if use_test:
        return matches
    else:
        return matches[:-size]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PO4ISR')
    seeds = [0, 10, 42, 625, 2023]

    train = []
    val = []
    test = []
    data_path = f"./final_dataset/LLM/total/"
    for seed in seeds:
        with open(f'{data_path}train_seed_{seed}_size_20.json', 'r') as json_file:
            data = json.load(json_file)
        train += data

        with open(f'{data_path}val_seed_{seed}_size_20.json', 'r') as json_file:
            data = json.load(json_file)
        val += data

        with open(f'{data_path}test_seed_{seed}_size_20.json', 'r') as json_file:
            data = json.load(json_file)
        test += data

    item_map = {}
    item_id = 0
    for data in train:
        item_list = extract_item(data['input'], 20, use_test=False)
        item_list.append(data['target'])
        for item in item_list:
            if item not in item_map.keys():
                item_map[item] = item_id
                item_id += 1

    for data in val:
        item_list = extract_item(data['input'], 20, use_test=True)
        item_list.append(data['target'])
        for item in item_list:
            if item not in item_map.keys():
                item_map[item] = item_id
                item_id += 1
    
    for data in test:
        item_list = extract_item(data['input'], 20, use_test=True)
        item_list.append(data['target'])
        for item in item_list:
            if item not in item_map.keys():
                item_map[item] = item_id
                item_id += 1
    
    print("size of item set:", len(item_map))
    np.save('./final_dataset/item_map_bundle.npy', item_map)