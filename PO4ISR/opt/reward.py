import numpy as np
from opt.utils import detect_error, extract_item_list

def ndcg(target_index):
    res = 1 / np.log2(target_index + 1)
    
    return res


class Reward():
    def __init__(self, config, request_model) -> None:
        self.reward_func = config['reward_func']
        self.request = request_model
  
    def calculate_reward(self, prompt, sample_data):
        reward = 0
        for data in sample_data:
            response = self.request.request(data['input'], prompt)
            if detect_error(response, data['target'], mode='select'):
                result_list = extract_item_list(response, data['target'])
                target_index = int(result_list[-1])
                reward = reward + ndcg(target_index)

        return reward