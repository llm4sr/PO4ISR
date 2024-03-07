import json
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PO4ISR')
    parser.add_argument('--seed', 
                        type=int,
                        default=42,
                        help='options: 42, 625, 2023, 0, 10')
    parser.add_argument('--candidate_size', 
                        type=int,
                        default=20,
                        help='the size of candidate size')
    
    args = parser.parse_args()

    save_path = './final_dataset/LLM/'
    dataset = ['food', 'electronic', 'clothing']
    train = []
    for data_name in dataset:
        with open(f'{save_path}{data_name}_train_seed_{args.seed}_size_{args.candidate_size}.json', 'r') as json_file:
            data = json.load(json_file)
        train += data
    
    val = []
    for data_name in dataset:
        with open(f'{save_path}{data_name}_val_seed_{args.seed}_size_{args.candidate_size}.json', 'r') as json_file:
            data = json.load(json_file)
        val += data
    
    test = []
    for data_name in dataset:
        with open(f'{save_path}{data_name}_test_seed_{args.seed}_size_{args.candidate_size}.json', 'r') as json_file:
            data = json.load(json_file)
        test += data

    save_path = f"./final_dataset/LLM/total/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f"{save_path}train_seed_{args.seed}_size_{args.candidate_size}.json", 'w') as f:
        json.dump(train, f)
    with open(f"{save_path}val_seed_{args.seed}_size_{args.candidate_size}.json", 'w') as f:
        json.dump(val, f)
    with open(f"{save_path}test_seed_{args.seed}_size_{args.candidate_size}.json", 'w') as f:
        json.dump(test, f)