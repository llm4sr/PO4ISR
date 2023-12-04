import torch
import numpy as np

def accuracy_calculator(rank_list, last, kpis):
    batch_size, topk = rank_list.size()
    expand_target = (last.squeeze()).unsqueeze(1).expand(-1, topk)
    hr = (rank_list == expand_target)
    
    ranks = (hr.nonzero(as_tuple=False)[:,-1] + 1).float()
    mrr = torch.reciprocal(ranks) # 1/ranks
    ndcg = 1 / torch.log2(ranks + 1)

    metrics = {
        'hr': hr.sum(axis=1).double().mean().item(),
        'mrr': torch.cat([mrr, torch.zeros(batch_size - len(mrr))]).mean().item(),
        'ndcg': torch.cat([ndcg, torch.zeros(batch_size - len(ndcg))]).mean().item()
    }

    
    return [metrics[kpi] for kpi in kpis]