import random
from math import sqrt

class SessionKNN(object):
    def __init__(self, train, test, params, logger):     
        self.neibor = params['n']
        self.train_data = train
        self.test_data = test
        self.sample_size = 1000
        self.sampling = 'random'
        self.logger = logger

        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict()
        self.item_session_map = dict()
        self.relevant_sessions = set()
        
        self.idx_session = {}


    def fit(self, train, valid_loader=None):
        for idx, items in enumerate(train):
            self.idx_session[idx] = items

        for idx, items in self.idx_session.items():
            self.session_item_map[idx] = set(items)

        for idx, items in self.idx_session.items():
            for item in items:
                if item in self.item_session_map:
                    self.item_session_map[item].add(idx)
                else:
                    self.item_session_map[item] = set([idx])


    def predict(self, test, k=15, candidate=None):
        preds = []
        last_items = []
        for idx, items in enumerate(test):
            item_set = items[:-1]
            item = items[-2]
            last_item = items[-1]

            if item not in self.item_session_map:
                preds.append([0] * k)
                last_items.append(last_item)
                continue
            else:
                neighbors = self.find_neighbors(set(item_set), item, idx)
                scores = self.score_items(neighbors)
                scores = sorted(scores, key=lambda x: scores[x], reverse=True)
                sub_scores = []
                for cand_item in candidate[idx]:
                    if cand_item in scores:
                        sub_scores.append(cand_item)
                if len(sub_scores) < k:
                    sub_scores = sub_scores + [0] * (k - len(sub_scores))
                else:
                    sub_scores = sub_scores[:k]
                preds.append(sub_scores)
                last_items.append(last_item)

        
        return preds, last_items
    
    def find_neighbors(self, session_items, item_id, session_idx):
        
        possible_neighbors = self.possible_neighbor_sessions(session_items, item_id, session_idx)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)

        possible_neighbors = sorted(possible_neighbors, key=lambda x: x[1], reverse=True)
        possible_neighbors = possible_neighbors[:self.neibor]

        return possible_neighbors
    
    def possible_neighbor_sessions(self, session_items, item_id, session_idx):
        
        self.relevant_sessions = self.relevant_sessions | self.item_session_map[item_id]
        if self.sample_size == 0:
            print('sample size is 0')
            return self.relevant_sessions
        else:
            self.relevant_sessions = self.relevant_sessions | self.item_session_map[item_id]
            if len(self.relevant_sessions) > self.sample_size:
                if self.sampling == 'recent':
                    print('recent')
                    # sample = self.most_recent_sessions(self.relevant_sessions, session_idx)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'item-item':
                    sample = self.item_item_sessions(session_items, session_idx)
                else:
                    sample = self.relevant_sessions[:self.sample_size]
                return sample
            else:
                return self.relevant_sessions
    

    def calc_similarity(self, session_items, sessions):
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt += 1
            session_items_test = self.session_item_map[session]
            similarity = self.session_similarity(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))
        return neighbors

    
    def session_similarity(self, first, second):
        # consine similarity
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result
    
    def score_items(self, neighbors):
        scores = dict()
        for session, score in neighbors:
            items = self.session_item_map[session]
            for item in items:
                if item in scores:
                    scores[item] += score
                else:
                    scores[item] = score
        return scores