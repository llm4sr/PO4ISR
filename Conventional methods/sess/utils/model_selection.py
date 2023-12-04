import numpy as np

def handle_adj(adj_dict, n_entity, sample_num, num_dict=None):
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])

    return adj_entity, num_entity


def build_graph(traindata, conf, model_conf):
    num = conf['num_node'] + 1
    seq = traindata
    sample_num = model_conf['n_sample']
    relation = []
    neighbor = [] * num

    all_test = set()

    adj1 = [dict() for _ in range(num)]
    adj = [[] for _ in range(num)]

    for i in range(len(seq)):
        data = seq[i]
        for k in range(1, 4):
            for j in range(len(data)-k):
                relation.append([data[j], data[j+k]])
                relation.append([data[j+k], data[j]])

    for tup in relation:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1

    weight = [[] for _ in range(num)]

    for t in range(num):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x]
        weight[t] = [v[1] for v in x]

    for i in range(num):
        adj[i] = adj[i][:sample_num]
        weight[i] = weight[i][:sample_num]
        
    return adj, weight