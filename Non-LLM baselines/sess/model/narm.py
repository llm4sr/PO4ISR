import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

epsilon = 1e-4

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    
    return dist
    
class NARM(nn.Module):
    def __init__(self, n_items, params, logger):
        '''
        NARM model class: https://dl.acm.org/doi/pdf/10.1145/3132847.3132926

        Parameters
        ----------
        n_items : int
            the number of items
        embedding_item_dim : int
            the dimension of item embedding
        hidden_size : int
            the hidden size of gru
        lr : float
            learning rate
        l2 : float
            L2-regularization term
        lr_dc_step : int
            Period of learning rate decay
        lr_dc : float
            Multiplicative factor of learning rate decay, by default 1 0.1
        n_layers : int, optional
            the number of gru layers, by default 1
        '''        
        super(NARM, self).__init__()
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.logger = logger
        # parameters
        self.n_items = n_items + 1 # 0 for None, so + 1
        self.embedding_item_dim = params['item_embedding_dim']
        self.hidden_size = params['hidden_size']
        self.n_layers = params['n_layers']
        # Embedding layer
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_item_dim, padding_idx=0)
        # Dropout layer
        self.emb_dropout = nn.Dropout(0.25)
        self.ct_dropout = nn.Dropout(0.5)
        # GRU layer
        self.gru = nn.GRU(self.embedding_item_dim, self.hidden_size, self.n_layers)
        # Linear layer
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.b = nn.Linear(self.embedding_item_dim, self.hidden_size, bias=False)
        self.sf = nn.Softmax(dim=1) #nn.LogSoftmax(dim=1)
        
        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['lr_dc_step'], gamma=params['lr_dc'])

        gpuid = params['gpu']
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        hidden = self.init_hidden(batch_size)
        embs = self.emb_dropout(self.item_embedding(seq))
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device = self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

#        c_t = torch.cat([c_local, c_global], 1)
#        c_t = self.ct_dropout(c_t)
        
        item_embs = self.item_embedding(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(c_global, self.b(item_embs).permute(1, 0)) # batch_size * item_size
        item_scores = self.sf(scores)
        
        return item_scores

    def fit(self, train_loader, validation_loader=None):
        gpuid = int(self.device.index)
        self.cuda(gpuid) if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')
        last_loss = 0.0
        for epoch in range(1, self.epochs + 1):  
            self.train()          
            total_loss = []
            for i, (seq, target, lens, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                scores = self.forward(seq.to(self.device), lens)
                loss = self.loss_function(torch.log(scores.clamp(min=1e-9)), target.squeeze().to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())

            delta_loss  = np.mean(total_loss) - last_loss
            if abs(delta_loss) < 1e-5:
                self.logger.info(f'early stop at epoch {epoch}')
                break
            last_loss = np.mean(total_loss)


            s = ''
            if validation_loader:
                valid_loss = self.evaluate(validation_loader)
                s = f'\tValidation Loss: {valid_loss:.4f}'
            self.logger.info(f'training epoch: {epoch}\tTrain Loss: {np.mean(total_loss):.3f}' + s)


    def predict(self, test_loader, k=15):
        self.eval()  
        preds, last_item = torch.tensor([]), torch.tensor([])
        for _, (seq, target_item, lens, candidate_set) in enumerate(test_loader):
            scores = self.forward(seq.to(self.device), lens)
            scores = torch.gather(scores, 1, candidate_set.to(self.device)).topk(k)[1]
            rank_list = torch.gather(candidate_set.to(self.device), 1, scores)
        
            preds = torch.cat((preds, rank_list.cpu()), 0)
            last_item = torch.cat((last_item, target_item), 0)

        return preds, last_item
        
    def mmr_rerank(self, test_loader, item_cate_matrix):
        itemnum = self.n_items # with padding
        preds, last_item = torch.tensor([]), torch.tensor([])
        # here set batch_size for test_loader is 1
        
        items_dist = euclidean_dist(torch.FloatTensor(item_cate_matrix), torch.FloatTensor(item_cate_matrix))
        
        for _, (seq, target_item, lens) in enumerate(test_loader):
            item_idx = list(range(1, itemnum))
            
            scores = self.forward(seq.to(self.device), lens)
            predictions = -scores[0, 1:] # - for 1st argsort DESC (101)
            rank = predictions.argsort().argsort()
            rank_list = []
            first = torch.LongTensor(item_idx)[rank==0].item()
            rank_list.append(first)
            item_idx.remove(first)
            predictions_candicate = predictions[rank!=0]
            
            for k in range(20-1):
            
                distance_toRL = torch.zeros((len(item_idx), len(rank_list))) #candidates * |RL|

                for i in range(len(item_idx)):
                    for j in range(len(rank_list)):
                        distance_toRL[i, j] = items_dist[item_idx[i], rank_list[j]]
                
                sum_bycolumn = distance_toRL.sum(dim=0).unsqueeze(1)
                min_distance_toRL = distance_toRL.min(dim=1)
                sum_respectiveIndices = sum_bycolumn[min_distance_toRL.indices.long()].squeeze()
                norm_distance_toRL = min_distance_toRL.values/sum_respectiveIndices
                with_mrr = torch.nn.Softmax(dim=0)(predictions_candicate) - 0.000005 * norm_distance_toRL.to(self.device)
                rank = with_mrr.argsort().argsort()
                first = torch.LongTensor(item_idx)[rank==0].item()
                rank_list.append(first)
                item_idx.remove(first)
                predictions_candicate = predictions_candicate[rank!=0]

            preds = torch.cat((preds, torch.tensor(rank_list).unsqueeze(0)), 0)
            last_item = torch.cat((last_item, target_item), 0)
            
        
        return preds, last_item

    def mmr_rerank_quick(self, test_loader, item_cate_matrix, lamb1):
        itemnum = self.n_items # with padding
        preds, last_item = torch.tensor([]), torch.tensor([])
        # here set batch_size for test_loader is 1
        
        items_dist = euclidean_dist(torch.FloatTensor(item_cate_matrix), torch.FloatTensor(item_cate_matrix))
        
        for _, (seq, target_item, lens) in enumerate(test_loader):
            item_idx = list(range(1, itemnum))
            all_candidates = list(range(1, itemnum))
            
            scores = self.forward(seq.to(self.device), lens)
            predictions = -scores[0, 1:] # - for 1st argsort DESC (101)
            rank = predictions.argsort().argsort()
            
            first = torch.LongTensor(all_candidates)[rank==0].item()
            rank_list = []
            rank_list.append(first)
            
            item_idx = torch.LongTensor(all_candidates)[rank<100].tolist()
            item_idx.remove(first)
            predictions_candicate = predictions[(rank!=0) & (rank<100)]
            
            for k in range(20-1):
            
                distance_toRL = torch.zeros((len(item_idx), len(rank_list))) #candidates * |RL|

                for i in range(len(item_idx)):
                    distance_toRL[i, :] = items_dist[item_idx[i]][rank_list]
                
                sum_bycolumn = distance_toRL.sum(dim=0).unsqueeze(1)
                min_distance_toRL = distance_toRL.min(dim=1)
                sum_respectiveIndices = sum_bycolumn[min_distance_toRL.indices.long()].squeeze()
                norm_distance_toRL = min_distance_toRL.values/sum_respectiveIndices
                with_mrr = torch.nn.Softmax(dim=0)(predictions_candicate) - lamb1 * norm_distance_toRL.to(self.device) # 0.000005
                rank = with_mrr.argsort().argsort()
                first = torch.LongTensor(item_idx)[rank==0].item()
                rank_list.append(first)
                item_idx.remove(first)
                predictions_candicate = predictions_candicate[rank!=0]

            preds = torch.cat((preds, torch.tensor(rank_list).unsqueeze(0)), 0)
            last_item = torch.cat((last_item, target_item), 0)
            
        
        return preds, last_item

    def evaluate(self, validation_loader):
        self.eval()
        valid_loss = []
        for _, (seq, target_item, lens) in enumerate(validation_loader):
            scores = self.forward(seq.to(self.device), lens)
            tmp_loss = self.loss_function(torch.log(scores.clamp(min=1e-9)), target_item.squeeze().to(self.device))
            valid_loss.append(tmp_loss.item())
            # TODO other metrics

        return np.mean(valid_loss)
        

