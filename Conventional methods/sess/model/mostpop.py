import torch

class MostPop(object):
    def __init__(self, pop_n, logger):
        '''

        Parameters
        ----------
        pop_n : int
            Only give back non-zero scores to the top N ranking items. 
            Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
        '''        
        self.top_n = pop_n
        # self.item_key = conf['item_key']
        # self.session_key = conf['session_key']
        self.logger = logger
        self.pop_list = None

    def fit(self, train, valid_loader=None):
        cnt_item = {}
        for idx, items in enumerate(train):
            for item in items:
                if item not in cnt_item:
                    cnt_item[item] = 1
                else:
                   cnt_item[item] += 1
        self.pop_list = sorted(cnt_item, key=lambda x: cnt_item[x], reverse=True)[:self.top_n]

    def predict(self, test, k=15):
        preds, last_item = torch.LongTensor([]), torch.LongTensor([])
        for items in test:
            seq = items[:-1]
            target = [items[-1]]
            pred = torch.LongTensor([self.pop_list[:k]])
            preds = torch.cat((preds, pred), 0)
            last_item = torch.cat((last_item, torch.tensor(target)), 0)
        # for seq, target, _ in test:
        #     pred = torch.LongTensor([self.pop_list.index[:k].tolist()])
        #     preds = torch.cat((preds, pred), 0)
        #     last_item = torch.cat((last_item, torch.tensor(target)), 0)

        return preds, last_item