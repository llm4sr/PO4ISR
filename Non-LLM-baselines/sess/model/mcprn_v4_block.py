import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm

epsilon = 1e-4

class PSRU(Module):
    def __init__(self, hidden_size):
        super(PSRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))

    def PSRUCell(self, inputs, hidden, concen):
        '''
        concen: batch_size
        '''
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        eps = 0.01
        concen_factor = (concen>=eps).to(torch.long).mul(concen).unsqueeze(1)
        concen_gate = concen_factor.mul(inputgate)
        hy = hidden - concen_gate * (hidden - newgate)
        return hy

    def forward(self, all_inputs, hidden, all_concen):
        '''
        all_inputs: batch_size * seq_len * dim
        hidden: batch_size * dim
        all_concen: batch_size * seq_len
        '''
        seq_len = all_inputs.shape[1]
        for i in range(seq_len):
            hidden = self.PSRUCell(all_inputs[:, i, :], hidden, all_concen[:, i])
        return hidden

        
class MCPRN(Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        embedding_dim(int): the dimension of item embedding
        purposes(int): the number of latent purposes

        hidden_size(int): the hidden size of gru
        
        batch_size(int):
        n_layers(int): the number of gru layers

    """
    def __init__(self, n_items, params, logger):
        super(MCPRN, self).__init__()
        self.n_items = n_items + 1
        self.epochs = params['epochs']
        self.embedding_dim = params['item_embedding_dim']
        self.hidden_size = params['item_embedding_dim']
        self.purposes = params['purposes']
        self.logger = logger
        self.tau = params['tau'] # hyperparameter addition
        self.batch_size = params['batch_size']
        
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)
        self.embPurpose = nn.Embedding(self.purposes, self.embedding_dim)

        # #purposes-channels
        self.purpose_layers = torch.nn.ModuleList()
        for _ in range(self.purposes):
            psru = PSRU(self.embedding_dim)
            self.purpose_layers.append(psru)
            
        
        self.cen_soft = nn.Softmax(dim=2)
        self.sf = nn.Softmax(dim=1)
        self.loss_function = nn.NLLLoss()
        self.reset_parameters()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['lr_dc_step'], gamma=params['lr_dc'])
        gpuid = params['gpu']
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, seq, target=None):
        seq = seq.transpose(0, 1) # batch_size * seqlen
        mask = (seq!=0).to(torch.long) # batch_size * seqlen
        

        seq_embs = self.emb(seq) # batch_size * seqlen * dim
        purpose_embs = self.embPurpose(torch.arange(self.purposes).to(self.device)) # #purposes * dim
        
        # Purpose Router
        concen_score = seq_embs.matmul(purpose_embs.t()) # batch_size * seqlen * #purposes
        concen_weight = self.cen_soft(concen_score/self.tau)
        concen_weight_real = mask.unsqueeze(2).mul(concen_weight)
        
        # MCPN
        batch_size = seq.size(0)
        hidden = self.init_hidden(batch_size)
        
        hn_list = []
        for i in range(self.purposes):
            hn = self.purpose_layers[i](seq_embs, hidden, concen_weight_real[:,:,i])# batch_size * dim
            hn_list.append(hn.unsqueeze(0))
            
        hn_embeddings = torch.cat(hn_list) # #purposes * batch_size * dim
        assert hn_embeddings.shape == (self.purposes, batch_size, self.embedding_dim)
        
        target = torch.arange(self.n_items).to(self.device) if isinstance(target, type(None)) else target
        pos_embs = self.emb(target.squeeze()) # batch_size(#items) * dim
        target_concen = pos_embs.matmul(purpose_embs.t()) # batch_size(#items) * #purposes
        target_concen_weight = nn.Softmax(dim=1)(target_concen)
        
        context_hn_list = []
        for i in range(self.purposes):
            context_hn = hn_embeddings[i].unsqueeze(1).expand(-1, target.shape[0], -1).mul(target_concen_weight[:, i].unsqueeze(1)) # batch_size * batch_size(#items) * dim
            context_hn_list.append(context_hn.unsqueeze(0))
        context_hn_embeddings = torch.cat(context_hn_list)
        assert context_hn_embeddings.shape == (self.purposes, batch_size, target.shape[0], self.embedding_dim)
        context_embedding = context_hn_embeddings.sum(axis=0) # batch_size * batch_size(#items) * dim
        
        try:
            pos_inner = torch.bmm(context_embedding, pos_embs.t().unsqueeze(0).expand(batch_size, -1, -1))
        except Exception as e:
            print(batch_size)
            print(context_embedding.shape)
            print(pos_embs.shape)
            print(e)
            assert 1!=1
        pos_inner_valid = torch.diagonal(pos_inner, dim1=1, dim2=2)# batch_size * #items
        
        #scores = torch.sigmoid(pos_inner_valid)
        scores = pos_inner_valid#self.sf(pos_inner_valid)
        assert scores.shape == (batch_size, target.shape[0])
        
        return scores


    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), requires_grad=True).to(self.device)

    def fit(self, train_loader, validation_loader=None):
        gpuid = int(self.device.index)
        self.cuda(gpuid) if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')
        last_loss = 0.0
        # train_loader = torch.utils.data.DataLoader(train_loader, batch_size=self.batch_size, shuffle=True)
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []
            for i, (seq, target, lens, _) in tqdm(enumerate(train_loader)):
            # for i, (seq, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                scores = self.forward(seq.to(self.device), target.to(self.device))
                # batch_size * batch_size
                loss = self.loss_function(torch.log(self.sf(scores).clamp(min=1e-9)), torch.arange(seq.size(1)).to(self.device))
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
        # test_loader = torch.utils.data.DataLoader(test_loader, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            preds, last_item = torch.tensor([]), torch.tensor([])
            for _, (seq, target_item, lens, candidate_set) in enumerate(test_loader):
            # for _, (seq, target_item) in enumerate(test_loader):    
                batch = 10000
                blocks = self.n_items//batch + 1
                score_blocks = []
                for i in range(blocks):
                    indices = torch.arange(i * batch, min((i + 1) * batch, self.n_items))
                    scores = self.forward(seq.to(self.device), indices.to(self.device))
                    score_blocks.append(scores)
                all_scores = torch.cat(score_blocks, 1)
                batch_size = seq.size(1)
                assert all_scores.shape == (batch_size, self.n_items)
                # all_scores = torch.gather(all_scores, 1, candidate_set.to(self.device))
                # rank_list = (torch.argsort(all_scores[:,1:], descending=True) + 1)[:,:k]  # why +1: +1 to represent the actual code of items
                
                all_scores = torch.gather(all_scores, 1, candidate_set.to(self.device)).topk(k)[1]
                rank_list = torch.gather(candidate_set.to(self.device), 1, all_scores)
                
                preds = torch.cat((preds, rank_list.cpu()), 0)
                last_item = torch.cat((last_item, target_item), 0)

        return preds, last_item

    def evaluate(self, validation_loader):
        self.eval()
        valid_loss = []
        # validation_loader = torch.utils.data.DataLoader(validation_loader, batch_size=self.batch_size, shuffle=False)
        for _, (seq, target_item, lens) in enumerate(validation_loader):
        # for _, (seq, target_item) in enumerate(validation_loader):
            scores = self.forward(seq.to(self.device))
            tmp_loss = self.loss_function(torch.log(scores.clamp(min=1e-9)), target_item.squeeze().to(self.device))
            valid_loss.append(tmp_loss.item())
            # TODO other metrics

        return np.mean(valid_loss)
        

