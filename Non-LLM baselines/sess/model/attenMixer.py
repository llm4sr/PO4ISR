import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


# requisite class
class LastAttenion(Module):

    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_lp_pool=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_lp_pool = use_lp_pool
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):

        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1),
                                          self.hidden_size // self.heads)  # batch_size x seq_length x latent_size
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        assert not torch.isnan(q0).any()
        assert not torch.isnan(q1).any()
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        assert not torch.isnan(alpha).any()
        if self.use_lp_pool == True:
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.last_layernorm(a)
        return a, alpha


class SessionGraphAttn(Module):
    def __init__(self, opt, n_node):
        super(SessionGraphAttn, self).__init__()
        self.hidden_size = opt['item_embedding_dim']
        self.n_node = n_node
        self.norm = opt['norm']
        self.scale = opt['scale']
        self.batch_size = opt['batch_size']
        self.heads = opt['heads']
        self.use_lp_pool = opt['use_lp_pool']
        self.softmax = opt['softmax']
        self.dropout = opt['dropout']
        self.last_k = opt['last_k']
        self.dot = opt['dot']
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.l_p = opt['l_p']
        self.mattn = LastAttenion(self.hidden_size, self.heads, self.dot, self.l_p, last_k=self.last_k,
                                  use_lp_pool=self.use_lp_pool)
        self.linear_q = nn.ModuleList()
        for i in range(self.last_k):
            self.linear_q.append(nn.Linear((i + 1) * self.hidden_size, self.hidden_size))

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get(self, i, hidden, alias_inputs):
        return hidden[i][alias_inputs[i]]

    def compute_scores(self, hidden, mask):

        hts = []

        lengths = torch.sum(mask, dim=1)

        for i in range(self.last_k):
            hts.append(self.linear_q[i](torch.cat(
                [hidden[torch.arange(mask.size(0)).long(), torch.clamp(lengths - (j + 1), -1, 1000)] for j in
                 range(i + 1)], dim=-1)).unsqueeze(1))

        ht0 = hidden[torch.arange(mask.size(0)).long(), torch.sum(mask, 1) - 1]

        hts = torch.cat(hts, dim=1)
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)

        hidden1 = hidden
        hidden = hidden1[:, :mask.size(1)]

        ais, weights = self.mattn(hts, hidden, mask)
        a = self.linear_transform(torch.cat((ais.squeeze(), ht0), 1))

        b = self.embedding.weight[1:]

        if self.norm:
            a = a.div(torch.norm(a, p=2, dim=1, keepdim=True) + 1e-12)
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
        b = F.dropout(b, self.dropout, training=self.training)
        scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 16 * scores
        return scores

    def forward(self, inputs):

        hidden = self.embedding(inputs)

        if self.norm:
            hidden = hidden.div(torch.norm(hidden, p=2, dim=-1, keepdim=True) + 1e-12)

        hidden = F.dropout(hidden, self.dropout, training=self.training)

        return hidden
        
# requisite end

class AreaAttnModel(Module):

    def __init__(self, opt, n_node, logger=None):
        super().__init__()

        self.opt = opt
        self.cnt = 0
        self.best_res = [0, 0]
        self.model = SessionGraphAttn(opt, n_node)
        self.loss = nn.Parameter(torch.Tensor(1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.opt['learning_rate'], weight_decay=self.opt['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt['lr_dc_step'], gamma=self.opt['lr_dc'])
        self.logger = logger
        gpuid = opt['gpu']
        self.devices = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    def forward(self, *args):

        return self.model(*args)

    
    def _forward(self, data):
        alias_inputs, A, items, mask, mask1, targets, n_node, candidate_set = data
        alias_inputs = alias_inputs.squeeze_().to(self.devices)
        A = A.squeeze_().to(self.devices)
        items = items.squeeze_().to(self.devices)
        mask = mask.squeeze_().to(self.devices)
        mask1 = mask1.squeeze_().to(self.devices)
        targets.squeeze_()
        n_node = n_node.squeeze_().to(self.devices)
        candidate_set = candidate_set.squeeze_().to(self.devices)

        hidden = self(items)

        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)

        if self.opt['norm']:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt['item_embedding_dim'])
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        return targets, scores, candidate_set


    def fit(self, train_data, validation_data=None):
        gpuid = int(self.devices.index)
        self.cuda(gpuid) if torch.cuda.is_available() else self.cpu()
        # self.cuda(1) if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')

        last_loss = 0.
        for epoch in range(1, self.opt['epochs'] + 1):
            self.train()
            total_loss = []
            current_loss = 0.
            train_loader = train_data
            for data in tqdm(train_loader):
                self.optimizer.zero_grad()
                targets, scores, _ = self._forward(data)
                targets = targets.long().to(self.devices)
                loss = self.model.loss_function(scores, targets - 1)
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
                # return loss
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
    
    def evaluate(self, validation_data):
        self.eval()
        valid_loss = []
        valid_loader = torch.utils.data.DataLoader(validation_data, num_workers=4, batch_size=self.batch_size,
                                                   shuffle=False, pin_memory=True)
        for data in tqdm(valid_loader):
            targets, scores = self._forward(data)
            targets = targets.long().to(self.devices)
            loss = self.loss_function(scores, (targets - 1).squeeze())
            valid_loss.append(loss.item())
        return np.mean(valid_loss)

    def predict(self, test_data, k=15):
        self.logger.info('Start predicting...')
        self.eval()
        test_loader = test_data
        preds, last_item = torch.tensor([]), torch.tensor([])
        for data in test_loader:
            targets, scores, candidate_data = self._forward(data)
            sub_scores = torch.gather(scores, 1, candidate_data-1).topk(k)[1]
            sub_scores = torch.gather(candidate_data, 1, sub_scores)
            preds = torch.cat((preds, sub_scores.cpu()), 0)
            last_item = torch.cat((last_item, torch.tensor(targets)), 0)
        return preds, last_item
