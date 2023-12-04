import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import normal_
epsilon = 1e-4

class STAMP(nn.Module):
    def __init__(self, n_items, params, logger):
        '''

        '''        
        super(STAMP, self).__init__()
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.logger = logger
        # parameters
        self.n_items = n_items + 1 # 0 for None, so + 1
        self.embedding_size = params['item_embedding_dim']
        # Embedding layer
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.w1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w3 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.w0 = nn.Linear(self.embedding_size, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.embedding_size), requires_grad=True)
        self.mlp_a = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.mlp_b = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sf = nn.Softmax(dim=1) #nn.LogSoftmax(dim=1)
        
        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=params['lr_dc_step'], gamma=params['lr_dc'])

        gpuid = params['gpu']
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        # # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.002)
        elif isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.05)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
                
    def count_alpha(self, context, aspect, output):
        r"""This is a function that count the attention weights
        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]
        Returns:
            torch.Tensor:attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        output_3dim = output.repeat(1, timesteps).view(-1, timesteps, self.embedding_size)
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(self.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha
        
    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        seq = seq.transpose(0,1)
        item_seq_emb = self.item_embedding(seq) # [b, seq_len, emb]
#        last_inputs = self.gather_indexes(item_seq_emb, lengths - 1)
        lengths = torch.Tensor(lengths).to(self.device)
        item_last_click_index = lengths - 1
        item_last_click = torch.gather(seq, dim=1, index=item_last_click_index.unsqueeze(1).long()) # [b, 1]
        last_inputs = self.item_embedding(item_last_click.squeeze())# [b, emb]
        org_memory = item_seq_emb # [b, seq_len, emb]
        ms = torch.div(torch.sum(org_memory, dim=1), lengths.unsqueeze(1).float())# [b, emb]
        alpha = self.count_alpha(org_memory, last_inputs, ms) # [b, seq_len]
        vec = torch.matmul(alpha.unsqueeze(1), org_memory) # [b, 1, emb]
        ma = vec.squeeze(1) + ms # [b, emb]
        hs = self.tanh(self.mlp_a(ma))
        ht = self.tanh(self.mlp_b(last_inputs))
        seq_output = hs * ht
        item_embs = self.item_embedding(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(seq_output, item_embs.permute(1, 0))
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
            # scores = torch.gather(scores, dim=1, index=candidate_set.to(self.device))
            scores = torch.gather(scores, 1, candidate_set.to(self.device)).topk(k)[1]
            rank_list = torch.gather(candidate_set.to(self.device), 1, scores)
            # rank_list = (torch.argsort(scores[:,1:], descending=True) + 1)[:,:k]  # why +1: +1 to represent the actual code of items

            preds = torch.cat((preds, rank_list.cpu()), 0)
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
        

