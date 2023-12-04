from tqdm import tqdm
from opt.metrics import Metric
from opt.request import Request
from opt.utils import extract_item_list

class Eval():
    def __init__(self, config, data, text_table):
        self.conf = config
        self.requset = Request(config)
        self.data = data
        self.text_table = text_table
        self.error_list = []
        self.target_rank_list = []
    
    def run(self, prompt):
        self.normal_eval(prompt)
        metric = Metric(self.target_rank_list, self.conf)
        result = metric.run()

        return result, self.target_rank_list, self.error_list
    
    def record_error(self, data, response):
        tmp = {}
        tmp['response'] = response
        tmp['target'] = data['target']
        tmp['input'] = data['input']
        tmp['target_index'] = data['target_index']
        
        return tmp

    def normal_eval(self, prompt):
        for data in tqdm(self.data):
            for i in range(3):
                response = self.requset.request(user=data['input'], system=prompt)
                result_list = extract_item_list(response, data['target'])
                if not result_list:
                    continue
                elif ((int(result_list[-1])) < self.conf['candidate_size']+1) and (int(result_list[-1]))>0:
                    self.target_rank_list.append(int(result_list[-1]))
                    break
            self.text_table.add_data(data['input'], data['target'], response)
            if (not result_list) or (int(result_list[-1]) >= (self.conf['candidate_size']+1)):
                error = self.record_error(data, response)
                self.error_list.append(error)
                self.target_rank_list.append(self.conf['candidate_size']+1)
    