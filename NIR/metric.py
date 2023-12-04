import pandas as pd
import numpy as np

class Metric():
    def __init__(self, rank_list) -> None:
        self.rank_list = rank_list
    
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
            if element <= 20:
                count += 1
        res['#valid_data'] = np.array([count, 0, 0])
        
        return res