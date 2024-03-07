import json
import pandas as pd
import os
import random
import argparse

def get_candidate_set(session_id, user_session, session_item_save, N, target):
    user2session = user_session.groupby('user ID')['session ID'].apply(list).to_dict()
    session_id = int(session_id)
    similar_sessions = []
    for user_id, sessions in user2session.items():
        if session_id in sessions:
            similar_sessions.extend(sessions)
    filtered_items = session_item_save[~session_item_save['session ID'].isin(similar_sessions)]['item ID']
    filtered_items = filtered_items.tolist()
    random_items = random.sample(filtered_items, N-1)
    random_index = random.randint(0, N-1)
    random_items.insert(random_index, target)

    return random_items, random_index+1

def get_text_session(item_list, item_titles):
    text = []
    for item in item_list:
        text.append(item_titles[item_titles['item ID'] == item]['titles'].values[0])
    text_string = ", ".join([f'{index+1}."{item}"' for index, item in enumerate(text)])
    
    return text_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PO4ISR')
    parser.add_argument('--seed', 
                        type=int,
                        default=42,
                        help='options: 42, 625, 2023, 0, 10')
    parser.add_argument('--candidate_size', 
                        type=int,
                        default=10,
                        help='the size of candidate size')
    parser.add_argument('--dataset', 
                        type=str,
                        default='clothing',
                        help='options: clothing, electronic, food')
    
    args = parser.parse_args()

    instruction = "Based on the user's current session interactions, you need to answer the following subtasks step by step:\n" \
              "1. Discover combinations of items within the session, where the number of combinations can be one or more.\n" \
              "2. Based on the items within each combination, infer the user's interactive intent within each combination.\n" \
              "3. Select the intent from the inferred ones that best represents the user's current preferences.\n" \
              "4. Based on the selected intent, please rerank the $number$ items in the candidate set according to the possibility of potential user interactions and give the ranking result of $number$ items.\n" \
              "Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set, and the output format for the ranking must be [no.<candidate item>;]\n"

    random.seed(args.seed)
    session_item = pd.read_csv(f"./raw_data/{args.dataset}/session_item.csv")
    session_bundle = pd.read_csv(f"./raw_data/{args.dataset}/session_bundle.csv")
    item_titles = pd.read_csv(f"./raw_data/{args.dataset}/item_titles.csv")
    bundle_item = pd.read_csv(f"./raw_data/{args.dataset}/bundle_item.csv")
    user_session = pd.read_csv(f"./raw_data/{args.dataset}/user_session.csv")
    bundle_intent = pd.read_csv(f"./raw_data/{args.dataset}/bundle_intent.csv")

    save_session = session_bundle['session ID'].unique()
    session_item_save = session_item[session_item['session ID'].isin(list(save_session))]

    session_dict = {}
    for _, row in session_item_save.iterrows():
        session_id = str(row["session ID"])
        item_id = row["item ID"]
        
        if session_id in session_dict:
            session_dict[session_id].append(item_id)
        else:
            session_dict[session_id] = [item_id]
    

    data_dict = {}
    candidate_num = args.candidate_size
    for k, v in session_dict.items():
        tmp = {}
        session_id = k
        session = v[:-1]
        target = v[-1]
        candidate_list, target_index = get_candidate_set(session_id, user_session, session_item_save, candidate_num, target)
        tmp['session'] = session
        tmp['target'] = target
        tmp['candidate'] = candidate_list
        tmp['target_index'] = target_index
        
        data_dict[session_id] = tmp
    
    data_text_dict = {}
    instruction_text = instruction.replace("$category$", args.dataset)
    instruction_text = instruction_text.replace("$number$", str(candidate_num))
    save_session_id = []
    for k, v in data_dict.items():
        tmp_text = {}
        session_id = int(k)
        tmp = v
        if len(tmp['session']) <= 1:
            continue
        input_session_text = get_text_session(tmp['session'], item_titles)
        input_candidate_text = get_text_session(tmp['candidate'], item_titles)
        input_text = f"Current session interactions: [{input_session_text}]\nCandidate set: [{input_candidate_text}]"

        target_item = tmp["target"]
        target_item = item_titles[item_titles['item ID'] == target_item]['titles'].values[0]

        tmp_text["instruction"] = instruction_text
        tmp_text["input"] = input_text
        tmp_text["target"] = target_item
        tmp_text["target_index"] = tmp['target_index']
        data_text_dict[k] = tmp_text
        save_session_id.append(session_id)
    
    user_session_save = user_session[user_session['session ID'].isin(save_session_id)]
    user_session_sorted = user_session_save.sort_values(by="timestamp")

    total_rows = len(user_session_sorted)
    train_size = int(0.8 * total_rows)
    val_size = int(0.1 * total_rows)

    train_session_ids = user_session_sorted['session ID'][:train_size].tolist()
    val_session_ids = user_session_sorted['session ID'][train_size:train_size + val_size].tolist()
    test_session_ids = user_session_sorted['session ID'][train_size + val_size:].tolist()

    train_set = []
    val_set = []
    test_set = []

    sessions = user_session_sorted["session ID"].values
    for session in sessions:
        if session in train_session_ids:
            train_set.append(data_text_dict[str(session)])
        elif session in val_session_ids:
            val_set.append(data_text_dict[str(session)])
        elif session in test_session_ids:
            test_set.append(data_text_dict[str(session)])
    

    save_path = f"./final_dataset/LLM/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}{args.dataset}_train_seed_{args.seed}_size_{args.candidate_size}.json", 'w') as f:
        json.dump(train_set, f)
    with open(f"{save_path}{args.dataset}_val_seed_{args.seed}_size_{args.candidate_size}.json", 'w') as f:
        json.dump(val_set, f)
    with open(f"{save_path}{args.dataset}_test_seed_{args.seed}_size_{args.candidate_size}.json", 'w') as f:
        json.dump(test_set, f)

