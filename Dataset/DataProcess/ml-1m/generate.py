import numpy as np
import pandas as pd
import random
import json
import os
from tqdm import tqdm
from datetime import datetime

def last_index(lst, element):
    for i in range(len(lst)-1, -1, -1):
        if lst[i] == element:
            return i
    return None

def split_by_timestamp(timestamps):
    dates = [datetime.utcfromtimestamp(ts) for ts in timestamps]
    date_to_timestamps = {}
    for date, ts in zip(dates, timestamps):
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in date_to_timestamps:
            date_to_timestamps[date_str] = []
        date_to_timestamps[date_str].append(ts)
    same_day_timestamps = [timestamps for timestamps in date_to_timestamps.values() if len(timestamps) > 1]
    return same_day_timestamps

def custom_sort(key):
    return sessions_id[key]['time']

def reindex_data(datas, item_mapping):
    final = []
    for data in datas:
        session = data['session']
        reindex_session = [item_mapping[i] for i in session]
        final.append({'session': reindex_session, 'time': data['time']})
    
    return final

def get_candiate_set(session, target, candidate_size, item_set):
    interact_set = set(session + [target])
    candidate_pool = list(item_set - interact_set)
    candidate_set = random.sample(candidate_pool, candidate_size-1)
    random_index = random.randint(0, candidate_size-1)
    candidate_set.insert(random_index, target)
    
    return candidate_set

def get_text(item_lst, title):
    text = []
    for item in item_lst:
        text.append(title[title['movie_id']==item].values[0][1].split(' (')[0])
    return text

def construct_train_val_text(title, dataset_id, item_set, candidate_size):
    data_text_lst = []
    candidates = []
    for index, data in enumerate(dataset_id):
        session_text = get_text(data['session'][:-1], title)
        input_session_string = ",".join([f'{index+1}."{item}"' for index, item in enumerate(session_text)])
        target_item = data['session'][-1]
        target_item_text = title[title['movie_id']==target_item].values[0][1].split(' (')[0]
        candidate_text = []
        candiate_set = get_candiate_set(data['session'][:-1], target_item, candidate_size, item_set)
        candidates.append(candiate_set)
        candidate_text = get_text(candiate_set, title)
        input_candidate_string = ",".join([f'{index+1}."{item}"' for index, item in enumerate(candidate_text)])
        input_text = f"Current session interactions: [{input_session_string}]\nCandidate set: [{input_candidate_string}]"
        target_item_index = candiate_set.index(target_item) + 1
        temp = {'input':input_text, 'target':target_item_text, 'target_index':target_item_index}
        data_text_lst.append(temp)
    return data_text_lst, candidates


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    sample_num = 150
    rating = pd.read_csv(f'./raw_data/ratings.dat', sep='::', header=None, 
                         names=['user', 'item', 'rating', 'timestamp'], engine='python')
    title =  pd.read_csv(f'./raw_data/movies.dat', sep='::', header=None, 
                         names=['movie_id', 'title', 'genre'], engine='python', encoding='ISO-8859-1')
    
    filter_num = 5
    tmp1 = rating.groupby(['item'], as_index=False)['user'].count()
    tmp1.rename(columns={'user': 'cnt_user'}, inplace=True)
    rating = rating.merge(tmp1, on=['item'])
    rating = rating.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()
    rating.drop(['cnt_user'], axis=1, inplace=True)

    item_set = set(rating['item'])

    interaction_dicts = dict()
    for _, row in rating.iterrows():
        user_id = int(row['user'])
        item_id = int(row['item'])
        timestamp = int(row['timestamp'])
    
        if user_id not in interaction_dicts:
            interaction_dicts[user_id] = {
                'item_id':[],
                'timestamp':[],
                'item_title':[],
            }
        interaction_dicts[user_id]['item_id'].append(item_id)
        interaction_dicts[user_id]['timestamp'].append(timestamp)
        item_title = title[title['movie_id']==item_id]['title'].values[0].split(' (')[0]
        interaction_dicts[user_id]['item_title'].append(item_title)
    

    sessions_id = {}
    s_id = 0
    window_size = 20
    for user_id in tqdm(interaction_dicts):
        temp = zip(interaction_dicts[user_id]['item_id'], interaction_dicts[user_id]['timestamp'], interaction_dicts[user_id]['item_title'])
        temp = sorted(temp, key=lambda x: x[1])
        result = zip(*temp)
        item_id, timestamps, item_title = [list(_) for _ in result]
        session_timestamps = split_by_timestamp(timestamps)
        for session_timestamp in session_timestamps:
            start_idx = timestamps.index(session_timestamp[0])
            end_idx = last_index(timestamps, session_timestamp[-1])
            session_idx_lst = item_id[start_idx:end_idx+1]
            if len(session_idx_lst) > window_size:
                for i in range(len(session_idx_lst) - window_size + 1):
                    session_time = session_timestamp[i+window_size-1]
                    sessions_id[s_id] = {'session': session_idx_lst[i: i+window_size], 'time': session_time}
                    s_id += 1
            else:
                session_time = session_timestamp[-1]
                sessions_id[s_id] = {'session': session_idx_lst, 'time': session_time}
                s_id += 1
    
    sorted_keys = sorted(sessions_id, key=custom_sort)
    sorted_dict = {key: sessions_id[key] for key in sorted_keys}

    final_sessions_id = []
    for i, (k, v) in enumerate(sorted_dict.items()):
        if v['session'] != []:
            final_sessions_id.append(v)

    rating_id = rating.copy()
    rating_id['original_item'] = rating_id['item']
    rating_id['item'] = rating_id.groupby('item').ngroup()
    item_mapping = rating_id.set_index('original_item')['item'].to_dict()

    save_path = f"./final_dataset/ID/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = f"./final_dataset/LLM/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save('./final_dataset/item_map_ml-1m.npy', item_mapping)
    np.save('./final_dataset/ID/total_id_ml-1m.npy', final_sessions_id)

    train_id = final_sessions_id[:int(len(final_sessions_id)*0.8)]
    valid_id = final_sessions_id[int(len(final_sessions_id)*0.8):int(len(final_sessions_id)*0.9)]
    test_id = final_sessions_id[int(len(final_sessions_id)*0.9):]
    np.save(f'./final_dataset/ID/train_itemnum_{len(item_mapping)}_ml-1m.npy', train_id)

    train_sample_id = random.sample(train_id, sample_num)
    reindex_train_sample_id = reindex_data(train_sample_id, item_mapping)
    np.save(f'./final_dataset/ID/train_itemnum_{len(item_mapping)}_sample_{sample_num}_ml-1m.npy', reindex_train_sample_id)

    train_sample_text, _ = construct_train_val_text(title, train_sample_id, item_set, candidate_size=20)
    with open(f'./final_dataset/LLM/train_{sample_num}.json', 'w') as json_file:
        json.dump(train_sample_text, json_file)
    
    sample_num = 1000
    valid_sample_id = random.sample(valid_id, sample_num)
    reindex_valid_sample_id = reindex_data(valid_sample_id, item_mapping)
    np.save(f'./final_dataset/ID/valid_itemnum_{len(item_mapping)}_sample_{sample_num}_ml-1m.npy', reindex_valid_sample_id)

    valid_sample_text, valid_cand_id = construct_train_val_text(title, valid_sample_id, item_set, candidate_size=20)
    with open(f'./final_dataset/LLM/valid_{sample_num}.json', 'w') as json_file:
        json.dump(valid_sample_text, json_file)

    reindex_valid_cand_id = []
    for data in valid_cand_id:
        reindex_valid_cand_id.append([item_mapping[i] for i in data])
    np.save(f'./final_dataset/ID/valid_candidate_sample_1000_ml-1m.npy', reindex_valid_cand_id)

    valid_sample_100 = random.sample(valid_sample_text, 100)
    with open(f'./final_dataset/LLM/valid_100.json', 'w') as json_file:
        json.dump(valid_sample_100, json_file)

    sample_num = 1000
    test_sample_id = random.sample(test_id, sample_num)
    reindex_test_sample_id = reindex_data(test_sample_id, item_mapping)
    np.save(f'./final_dataset/ID/test_itemnum_{len(item_mapping)}_sample_{sample_num}_ml-1m.npy', reindex_test_sample_id)

    seeds = [0, 10, 42, 625, 2023]
    for seed in seeds:
        random.seed(seed)
        test_sample_text, test_cand_id = construct_train_val_text(title, test_sample_id, item_set, candidate_size=20)
        reindex_test_cand_id = []
        for data in test_cand_id:
            reindex_test_cand_id.append([item_mapping[i] for i in data])
        np.save(f'./final_dataset/ID/test_candidate_{seed}_ml-1m.npy', reindex_test_cand_id)
        
        with open(f'./final_dataset/LLM/test_seed_{seed}.json', 'w') as json_file:
            json.dump(test_sample_text, json_file)
    
    