import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import Module
import torch.nn.functional as F

class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output
        
# requisite end



class CombineGraph(Module):
    def __init__(self, conf, num_node, adj_all, num, logger):
    # conf is model_conf
        super(CombineGraph, self).__init__()
        self.dim = conf['item_embedding_dim']
        self.num_node = num_node
        self.batch_size = conf['batch_size']
        self.dropout_local = conf['dropout_local']
        self.dropout_global = conf['dropout_global']
        self.hop = conf['num_hop']
        self.sample_num = conf['n_sample']
        gpuid = conf['gpu']
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        self.adj_all = torch.Tensor(adj_all).long().to(self.device)
        self.num = torch.Tensor(num).float().to(self.device)

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, 0.2, dropout=0.0)
        self.global_agg = []
        for i in range(self.hop):
            if conf['activate'] == 'relu':
                agg = GlobalAggregator(self.dim, conf['dropout_gcn'], act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, conf['dropout_gcn'], act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=conf['learning_rate'], weight_decay=conf['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=conf['lr_dc_step'], gamma=conf['lr_dc'])
        self.reset_parameters()

        self.logger = logger
        self.epochs = conf['epochs']

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores
        
    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]
        
    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local = self.local_agg(h, adj, mask_item)

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        
        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop+1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        # combine
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        output = h_local + h_global

        return output

    def _forward(self, data):
        alias_inputs, adj, items, mask, targets, inputs, candidate_set = data
        alias_inputs = alias_inputs.long().to(self.device)
        items = items.long().to(self.device)
        adj = adj.float().to(self.device)
        mask = mask.long().to(self.device)
        inputs = inputs.long().to(self.device)
        candidate_set = candidate_set.long().to(self.device)

        hidden = self.forward(items, adj, mask, inputs)
        get = lambda index: hidden[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        return targets, self.compute_scores(seq_hidden, mask), candidate_set

    def fit(self, train_data, validation_data=None):
        gpuid = int(self.device.index)
        self.cuda(gpuid) if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')
        last_loss = 0.0
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []

            train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=self.batch_size,
                                                       shuffle=True, pin_memory=True)
            for data in tqdm(train_loader):
                self.optimizer.zero_grad()
                targets, scores, _ = self._forward(data)
                targets = targets.long().to(self.device)
                loss = self.loss_function(scores, (targets-1).squeeze())
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
            self.scheduler.step()
            
            current_loss = np.mean(total_loss)
            delta_loss = current_loss - last_loss
            if abs(delta_loss) < 1e-5:
                self.logger.info(f'Early stop at epoch {epoch}')
                break
            last_loss = current_loss

            s = ''
            if validation_data:
                valid_loss = self.evaluate(validation_data)
                s = f'\tValidation Loss: {valid_loss:.4f}'
            self.logger.info(f'Training Epoch: {epoch}\tLoss: {np.mean(total_loss):.4f}' + s)

    def predict(self, test_data, k=15):
        self.logger.info('Start predicting...')
        self.eval()
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=self.batch_size,
                                              shuffle=False, pin_memory=True)
        preds, last_item = torch.tensor([]), torch.tensor([])
        for data in test_loader:
            targets, scores, candidate_set = self._forward(data)
            sub_scores = torch.gather(scores, 1, candidate_set-1).topk(k)[1]
            # scores = torch.gather(scores, 1, candidate_set.to(self.device)).topk(k)[1]
            sub_scores = torch.gather(candidate_set, 1, sub_scores)
            # sub_scores = scores.topk(k)[1]
            # sub_scores = sub_scores + 1 # +1 change to actual code we did before

            preds = torch.cat((preds, sub_scores.cpu()), 0)
            last_item = torch.cat((last_item, torch.tensor(targets)), 0)

        return preds, last_item

    def evaluate(self, validation_data):
        self.eval()
        valid_loss = []
        valid_loader = torch.utils.data.DataLoader(validation_data, num_workers=4, batch_size=self.batch_size,
                                                   shuffle=False, pin_memory=True)
        for data in tqdm(valid_loader):
            targets, scores = self._forward(data)
            targets = targets.long().to(self.device)
            loss = self.loss_function(scores, (targets - 1).squeeze())
            valid_loss.append(loss.item())
        return np.mean(valid_loss)
