import numpy as np
import pandas as pd

class Metric():
    def __init__(self, rank_list, conf) -> None:
        self.rank_list = rank_list
        self.conf = conf
    
    def ndcg(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append((1 / np.log2(rank + 1)))
        
        return np.mean(res)
    
    def hit(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append(1)
        return np.mean(res)
    
    def map(self, N):
        res = []
        for rank in self.rank_list:
            if rank > N:
                res.append(0)
            else:
                res.append((1 / rank))
        return np.mean(res)
    
    def run(self):
        res = pd.DataFrame({'KPI@K': ['NDCG', 'HIT', 'MAP']})
        if self.conf['candidate_size'] == 10:
            topk_list = [1, 5, 10]
        elif self.conf['candidate_size'] == 20:
            topk_list = [1, 5, 10, 20]
        for topk in topk_list:
            metric_res = []
            metric_res.append(self.ndcg(topk))
            metric_res.append(self.hit(topk))
            metric_res.append(self.map(topk))

            metric_res = np.array(metric_res)
            res[topk] = metric_res
        count = 0
        for element in self.rank_list:
            if element <= self.conf['candidate_size']:
                count += 1
        res['#valid_data'] = np.array([count, 0, 0])
        
        return res
