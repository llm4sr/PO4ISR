import random
from opt.utils import detect_error, extract_edit_prompt

class Improve():
    def __init__(self,
                inferring_reasons, 
                refining_prompts, 
                augumenting_prompts, 
                train_data,
                config,
                request_model):
        self.inferring_reasons = inferring_reasons
        self.refining_prompts = refining_prompts
        self.augumenting_prompts = augumenting_prompts
        self.train_data = train_data
        self.config = config
        self.request = request_model
        self.used_data = []
    
    def evaluate_collect_error(self, prompt, data):
        errors_list = []
        for val in data:
            response = self.request.request(val['input'], prompt)
            if not detect_error(response, val['target']):
                error = {}
                error['input'] = val['input']
                error['output'] = response
                errors_list.append(error)
    
        return errors_list

    def generate_similar_prompt(self, prompt_list):
        similar_prompts = []
        for prompt in prompt_list:
            tmp = self.augumenting_prompts
            content = tmp.replace("$refined_prompt$", prompt)
            for i in range(self.config['addition_sample']):
                response = self.request.request(user=content, system='')
                similar_prompts.append(response)
    
        return similar_prompts

    def run(self, prompt, table=None):
        candidate_prompts = []
        batch_data = random.sample(self.train_data, self.config['batch_size'])
        self.used_data += batch_data
        errors_list = self.evaluate_collect_error(prompt, batch_data) 
        try:
            errors_group = random.sample(errors_list, self.config['error_batch_size'])
        except:
            errors_group = errors_list
        inferring_reasons = self.inferring_reasons.replace("$prompt$", prompt).replace("$num_feedbacks$", str(self.config['num_feedbacks'])) 
        refining_prompts = self.refining_prompts.replace("$prompt$", prompt)
        
        for error in errors_group:
            # Inferring reasons for errors
            tmp_prompt = inferring_reasons
            content = tmp_prompt.replace("$error_case$", error['input']) #error['output'] --> error['input']
            gradient = self.request.request(user=content, system='')

            # Refining prompts with reasons
            tmp_prompt = refining_prompts
            tmp_prompt = tmp_prompt.replace("$error_case$", error['input']) #error['output'] --> error['input']
            content = tmp_prompt.replace("$reasons$", gradient)
            edit_prompt = self.request.request(user=content, system='')
            edit_prompt_list = extract_edit_prompt(edit_prompt)

            # Augumenting prompts
            similar_prompts = self.generate_similar_prompt(edit_prompt_list)

            # Merge candidate prompts
            candidate_prompts.extend(edit_prompt_list)
            candidate_prompts.extend(similar_prompts)
            
            # add data into wandb Text Table [input, prompt, reason, improved prompt, augumented prompt]
            if self.config['use_wandb']:
                for new_index, new_prompt in enumerate(edit_prompt_list):
                    for mc_index in range(self.config['addition_sample']):
                        table.add_data(error['input'], prompt, gradient, new_prompt, similar_prompts[new_index * self.config['addition_sample'] + mc_index])
        # Randomly sampled #num successor candidates per parent prompt
        try:
            sample_candidate_prompts = random.sample(candidate_prompts, self.config['num_candidates'])
        except:
            sample_candidate_prompts = candidate_prompts
        return sample_candidate_prompts
    
    def get_used_data(self):
        return self.used_data