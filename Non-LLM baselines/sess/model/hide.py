import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse

# requisite class
class DisentangleGraph(nn.Module):
    def __init__(self, dim, alpha, e=0.3, t=10.0):
        super(DisentangleGraph, self).__init__()
        # Disentangling Hypergraph with given H and latent_feature
        self.latent_dim = dim   # Disentangled feature dim
        self.e = e              # sparsity parameters
        self.t = t              
        self.w = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
        self.w1 = nn.Parameter(torch.Tensor(self.latent_dim, 1))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, H, int_emb, mask):
        """
        Input: intent-aware hidden:(Batchsize, N, dim), incidence matrix H:(batchsize, N, num_edge), intention_emb: (num_factor, dim), node mask:(batchsize, N)
        Output: Distangeled incidence matrix
        """
        node_num = torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1) # (batchsize, 1, 1)
        select_k = self.e * node_num
        select_k = select_k.floor() 

        mask = mask.float().unsqueeze(-1) # (batchsize, N, 1)
        h = hidden
        batch_size = h.shape[0]
        N = H.shape[1]
        k = int_emb.shape[0]

        select_k = select_k.repeat(1, N, k)

          
        int_emb =  int_emb.unsqueeze(0).repeat(batch_size, 1, 1) # (batchsize, num_factor, latent_dim)
        int_emb =  int_emb.unsqueeze(1).repeat(1, N, 1, 1)       # (batchsize, N, num_factor, latent_dim)

        hs = h.unsqueeze(2).repeat(1, 1, k, 1)                   # (batchsize, N, num_factor, latent_dim)

        # CosineSimilarity 
        cos = nn.CosineSimilarity(dim=-1)
        sim_val = self.t * cos(hs, int_emb)                      # (batchsize, Node, Num_edge)
        
        
        sim_val = sim_val * mask
        
        # sort
        _, indices = torch.sort(sim_val, dim=1, descending=True)
        _, idx = torch.sort(indices, dim=1)

        # select according to <=0
        judge_vec = idx - select_k  
        ones_vec = 3*torch.ones_like(sim_val)
        zeros_vec = torch.zeros_like(sim_val)
        
        # intent hyperedges
        int_H = torch.where(judge_vec <= 0, ones_vec, zeros_vec)
        # add intent hyperedge
        H_out = torch.cat([int_H, H], dim=-1) # (batchsize, N, num_edge+1) 
        # return learned binary value
        return H_out


class LocalHyperGATlayer(nn.Module):
    def __init__(self, dim, layer, alpha, dropout=0., bias=False, act=True):
        super(LocalHyperGATlayer, self).__init__()
        self.dim = dim
        self.layer = layer
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        self.act = act

        if self.act:
            self.acf = torch.relu

        
        # Parameters 
        # node->edge->node
        self.w1 = Parameter(torch.Tensor(self.dim, self.dim))
        self.w2 = Parameter(torch.Tensor(self.dim, self.dim))
  
        self.a10 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))   
        self.a11 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))   
        self.a12 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))    
        self.a20 = nn.Parameter(torch.Tensor(size=(self.dim, 1))) 
        self.a21 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))     
        self.a22 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))  
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, hidden, H, s_c):
        """
        Input: hidden:(Batchsize, N, latent_dim), incidence matrix H:(batchsize, N, num_edge), session cluster s_c:(Batchsize, 1, latent_dim)
        Output: updated hidden:(Batchsize, N, latent_dim)
        """
        batch_size = hidden.shape[0]
        N = H.shape[1]            # node num
        edge_num = H.shape[2]     # edge num
        H_adj = torch.ones_like(H)
        mask = torch.zeros_like(H)
        H_adj = torch.where(H>0, H_adj, mask)
        s_c = s_c.expand(-1, N, -1)
        h_emb = hidden
        h_embs = []

        for i in range(self.layer):
            edge_cluster = torch.matmul(H_adj.transpose(1,2), h_emb)                  # (Batchsize, edge_num, latent_dim)
            h_t_cluster = h_emb + s_c
            
            # node2edge
            edge_c_in = edge_cluster.unsqueeze(1).expand(-1, N, -1, -1)               # (Batchsize, N, edge_num, latent_dim)
            h_4att0 = h_emb.unsqueeze(2).expand(-1, -1, edge_num, -1)                 # (Batchsize, N, edge_num, latent_dim)

            feat = edge_c_in * h_4att0

            atts10 = self.leakyrelu(torch.matmul(feat, self.a10).squeeze(-1))         # (Batchsize, N, edge_num)
            atts11 = self.leakyrelu(torch.matmul(feat, self.a11).squeeze(-1))         # (Batchsize, N, edge_num)
            atts12 = self.leakyrelu(torch.matmul(feat, self.a12).squeeze(-1))         # (Batchsize, N, edge_num)
            
            zero_vec = -9e15*torch.ones_like(H)
            alpha1 = torch.where(H.eq(1), atts10, zero_vec)
            alpha1 = torch.where(H.eq(2), atts11, alpha1)
            alpha1 = torch.where(H.eq(3), atts12, alpha1)

            alpha1 = F.softmax(alpha1, dim=1)                                         # (Batchsize, N, edge_num)

            edge = torch.matmul(alpha1.transpose(1,2), h_emb)                         # (Batchsize, edge_num, latent_dim)

            # edge2node
            edge_in = edge.unsqueeze(1).expand(-1, N, -1, -1)                         # (Batchsize, N, edge_num, latent_dim)
            h_4att1 = h_t_cluster.unsqueeze(2).expand(-1, -1, edge_num, -1)           # (Batchsize, N, edge_num, latent_dim)
            
            feat_e2n = edge_in * h_4att1
            
            atts20 = self.leakyrelu(torch.matmul(feat_e2n, self.a20).squeeze(-1))     # (Batchsize, N, edge_num)
            atts21 = self.leakyrelu(torch.matmul(feat_e2n, self.a21).squeeze(-1))     # (Batchsize, N, edge_num)
            atts22 = self.leakyrelu(torch.matmul(feat_e2n, self.a22).squeeze(-1))     # (Batchsize, N, edge_num)
            

            alpha2 = torch.where(H.eq(1), atts20, zero_vec)
            alpha2 = torch.where(H.eq(2), atts21, alpha2)
            alpha2 = torch.where(H.eq(3), atts22, alpha2)
            
            alpha2 = F.softmax(alpha2, dim=2)                                         # (Batchsize, N, edge_num)

            h_emb = torch.matmul(alpha2, edge)                                        # (Batchsize, N, latent_dim)
            h_embs.append(h_emb)

        h_embs = torch.stack(h_embs, dim=1)
        h_out = torch.sum(h_embs, dim=1)

        return h_out
        
# requisite end

class HIDE(Module):
    def __init__(self, conf, num_node, adj_all=None, num=None, cat=False, logger=None):
        super(HIDE, self).__init__()
        # HYPER PARA
        self.conf = conf 
        self.batch_size = conf['batch_size']
        self.num_node = num_node
        self.dim = conf['item_embedding_dim']
        self.dropout_local = conf['dropout_local']
        self.dropout_global = conf['dropout_global']
        self.sample_num = conf['n_sample']
        # self.nonhybrid = conf['nonhybrid']
        self.layer = int(conf['n_layers'])
        self.n_factor = conf['n_factor']    # number of intention prototypes
        self.cat = cat  # no use
        self.e = conf['e']
        self.disen = conf['disen']
        self.w_k = 10
        gpuid = conf['gpu']
        self.devices = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        self.logger = logger

        
        # Item representation
        self.embedding = nn.Embedding(num_node, self.dim)
        
        if self.disen:
            self.feat_latent_dim = self.dim // self.n_factor
            self.split_sections = [self.feat_latent_dim] * self.n_factor
            
        else:
            self.feat_latent_dim = self.dim
        
        # Position representation
        self.pos_embedding = nn.Embedding(200, self.dim)


        
        if self.disen:
            self.disenG = DisentangleGraph(dim=self.feat_latent_dim, alpha=self.conf['alpha'], e=self.e) # need to be updated
            self.disen_aggs = nn.ModuleList([LocalHyperGATlayer(self.feat_latent_dim, self.layer, self.conf['alpha'], self.conf['dropout_gcn']) for i in range(self.n_factor)])
        else:
            self.local_agg = LocalHyperGATlayer(self.dim, self.layer, self.conf['alpha'], self.conf['dropout_gcn'])



        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(3 * self.dim, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=True)
        
        
        self.leakyrelu = nn.LeakyReLU(self.conf['alpha'])
        # main task loss
        self.loss_function = nn.CrossEntropyLoss()
        if self.disen:
            # define for the additional losses
            self.classifier = nn.Linear(self.feat_latent_dim,  self.n_factor)
            self.loss_aux = nn.CrossEntropyLoss()
            self.intent_loss = 0

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.conf['learning_rate'], weight_decay=self.conf['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.conf['lr_dc_step'], gamma=self.conf['lr_dc'])

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_disentangle_loss(self, intents_feat):
        # compute discrimination loss
        
        labels = [torch.ones(f.shape[0])*i for i, f in enumerate(intents_feat)] # lable: 0, 1, ..., intent_num-1
        labels = torch.cat(tuple(labels), 0).long().to(self.devices)
        intents_feat = torch.cat(tuple(intents_feat), 0)

        pred = self.classifier(intents_feat)
        discrimination_loss = self.loss_aux(pred, labels)
        return discrimination_loss

    def compute_scores(self, hidden, mask, item_embeddings):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        ht = hidden[:, 0, :]
        ht = ht.unsqueeze(-2).repeat(1, len, 1)             # (b, N, dim)
        
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        

        hs = torch.cat([hs, ht], -1).matmul(self.w_s)

        feat = hs * hidden  
        nh = torch.sigmoid(torch.cat([self.glu1(nh), self.glu2(hs), self.glu3(feat)], -1))

        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        

        if self.disen:
            select = torch.sum(beta * hidden, 1)

            score_all = []
            select_split = torch.split(select, self.split_sections, dim=-1)
            b = torch.split(item_embeddings[1:], self.split_sections, dim=-1)
            for i in range(self.n_factor):
                sess_emb_int = self.w_k * select_split[i]
                item_embeddings_int = b[i]
                scores_int = torch.mm(sess_emb_int, torch.transpose(item_embeddings_int, 1, 0))
                score_all.append(scores_int)
            
            score = torch.stack(score_all, dim=1)   # (b ,k, item_num)
            scores = score.sum(1)

        else:
            select = torch.sum(beta * hidden, 1)
            b = item_embeddings[1:]  # n_nodes x latent_size
            scores = torch.matmul(select, b.transpose(1, 0))

        return scores


    def forward(self, inputs, Hs, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]

        item_embeddings = self.embedding.weight
        
        #zeros = torch.cuda.FloatTensor(1, self.dim).fill_(0)
        zeros = torch.FloatTensor(1, self.dim).fill_(0).to(self.devices)
        item_embeddings = torch.cat([zeros, item_embeddings], 0)

        h = item_embeddings[inputs]
        item_emb = item_embeddings[item] * mask_item.float().unsqueeze(-1)
 
        session_c = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        session_c = session_c.unsqueeze(1)  # (batchsize, 1, dim)
        
        
        if self.disen:
            # intent prototypes from the clustering of all items
            all_items = item_embeddings[1:]   # item_num x dim
            intents_cat = torch.mean(all_items, dim=0, keepdim=True) # 1 x dim
            # Parallel disen-encoders
            mask_node = torch.ones_like(inputs)
            zeor_vec = torch.zeros_like(inputs)
            mask_node = torch.where(inputs.eq(0), zeor_vec, mask_node)
            
            h_split = torch.split(h, self.split_sections, dim=-1)
            s_split = torch.split(session_c, self.split_sections, dim=-1)
            intent_split = torch.split(intents_cat, self.split_sections, dim=-1)
            h_ints = []
            intents_feat = []
            for i in range(self.n_factor):
                h_int = h_split[i]
                Hs = self.disenG(h_int, Hs, intent_split[i], mask_node)  #  construct intent hyperedges for each item ?
                h_int = self.disen_aggs[i](h_int, Hs, s_split[i])

                # Activate disentangle with intent protypes
                # better 
                intent_p = intent_split[i].unsqueeze(0).repeat(batch_size, seqs_len, 1)
                # CosineSimilarity
                sim_val = h_int * intent_p
                cor_att = torch.sigmoid(sim_val)
                h_int = h_int * cor_att + h_int

                
                h_ints.append(h_int)
                intents_feat.append(torch.mean(h_int, dim=1))   # (b ,latent_dim)
                
           
            h_stack = torch.stack(h_ints, dim=2)   # (b ,len, k, latent_dim)
            h_local = h_stack.reshape(batch_size, seqs_len, self.dim)

            # Aux task: intent prediction
            self.intent_loss = self.compute_disentangle_loss(intents_feat)

        else:

            h_local = self.local_agg(h, Hs, session_c)
                        
        
        output = h_local
               
        return output, item_embeddings

    def _forward(self, data):
        alias_inputs, Hs, items, mask, targets, inputs, candidate_set = data
        alias_inputs = alias_inputs.long().to(self.devices)
        items = items.long().to(self.devices)
        Hs = Hs.float().to(self.devices)
        mask = mask.long().to(self.devices)
        inputs = inputs.long().to(self.devices)
        candidate_set = candidate_set.long().to(self.devices)

        hidden, item_embeddings = self.forward(items, Hs, mask, inputs)
        get = lambda index: hidden[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        return targets, self.compute_scores(seq_hidden, mask, item_embeddings), candidate_set

    def fit(self, train_data, validation_data=None):
        gpuid = int(self.devices.index)
        self.cuda(gpuid) if torch.cuda.is_available() else self.cpu()
        self.logger.info('Start training...')

        last_loss = 0.
        for epoch in range(1, self.conf['epochs'] + 1):
            self.train()
            total_loss = []
            current_loss = 0.
            train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=self.batch_size,
                                                       shuffle=True, pin_memory=True)
            for data in tqdm(train_loader):
                self.optimizer.zero_grad()
                targets, scores, _ = self._forward(data)
                targets = targets.long().to(self.devices)
                loss = self.loss_function(scores, (targets-1).squeeze())
                if self.disen:
                    loss += self.conf['lamda'] * self.intent_loss
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
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=self.batch_size,
                                              shuffle=False, pin_memory=True)
        preds, last_item = torch.tensor([]), torch.tensor([])
        for data in test_loader:
            targets, scores, candidate_data = self._forward(data)
            sub_scores = torch.gather(scores, 1, candidate_data-1).topk(k)[1]
            sub_scores = torch.gather(candidate_data, 1, sub_scores)
            # sub_scores = scores.topk(k)[1]
            # sub_scores = sub_scores + 1

            preds = torch.cat((preds, sub_scores.cpu()), 0)
            last_item = torch.cat((last_item, torch.tensor(targets)), 0)
        return preds, last_item
