import random
import math

class Select():
    def __init__(self,
                train_data,
                config,
                reward_model):
        self.train_data = train_data
        self.config = config
        self.reward_model = reward_model
        self.used_data = []

    def ucb(self, prompt_list):
        numbers_of_selections = [0] * len(prompt_list)
        sums_of_reward = [0] * len(prompt_list)
        index_list = [i for i in range(len(prompt_list))]

        for t in range(1, self.config['time_steps']+1):
            sample_data = random.sample(self.train_data, self.config['sample_num'])
            self.used_data += sample_data
            if t == 1:
                select_prompt_index = random.choice(index_list)
            else:
                explore_param = self.config['explore_param']
                results = [q_value + explore_param*math.sqrt(math.log(t)/(n+1)) for q_value, n in zip(sums_of_reward, numbers_of_selections)]
                max_result = max(results)
                select_prompt_index = results.index(max_result)
            select_prompt = prompt_list[select_prompt_index]
            select_prompt_reward = self.reward_model.calculate_reward(select_prompt, sample_data)

            # Update N and Q
            numbers_of_selections[select_prompt_index] += self.config['sample_num']
            sums_of_reward[select_prompt_index] += select_prompt_reward / numbers_of_selections[select_prompt_index]

        # Return top b prompts
        if self.config['beam_width'] > len(prompt_list):
            raise Exception("The value of beamwidth needs to be less than the length of the prompt list")
        else:
            pairs = list(zip(sums_of_reward, prompt_list))
            pairs.sort(reverse=True)
            top_b_prompt = [pair[1] for pair in pairs[:self.config['beam_width']]]
        
        return top_b_prompt
    
    def run(self, prompt_list):
        top_b_prompt = self.ucb(prompt_list)

        return top_b_prompt
    
    def get_used_data(self):
        return self.used_data



    
        