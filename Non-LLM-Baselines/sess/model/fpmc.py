import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

class FPMC(nn.Module):
    def __init__(self, n_items, params, logger):
        super(FPMC, self).__init__()
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.embedding_size = params['item_embedding_dim']
        self.n_items = n_items + 1
        # last click item embedding matrix
        self.LI_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # label embedding matrix
        self.IL_emb = nn.Embedding(self.n_items, self.embedding_size)
        self.loss_func = BPRLoss()
        self.logger = logger
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'], weight_decay=params['l2'])
        gpuid = params['gpu']
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
            
    def forward(self, seq, lengths):
        batch_size = seq.size(1)
        seq = seq.transpose(0,1)
        item_last_click_index = (torch.LongTensor(lengths)-1).to(self.device)
        item_last_click = torch.gather(seq, dim=1, index=item_last_click_index.unsqueeze(1)) # [b, 1]
        item_seq_emb = self.LI_emb(item_last_click.squeeze())  # [b,emb]
        
        #il_emb = self.IL_emb(next_item) # TODO next_item
        #il_emb = torch.unsqueeze(il_emb, dim=1)  # [b,n,emb] in here n = 1
        il_emb = self.IL_emb(torch.arange(self.n_items).to(self.device)) # [b,n,emb] in here n = item_size

        # This is the core part of the FPMC model,can be expressed by a combination of a MF and a FMC model
        #  MF  # MF part is dropped here because of anonymous user
#        mf = torch.matmul(user_emb, iu_emb.permute(0, 2, 1))
#        mf = torch.squeeze(mf, dim=1)  # [B,1]
        #  FMC
        fmc = torch.matmul(item_seq_emb, il_emb.permute(1, 0)) # [b,n]
        #fmc = torch.squeeze(fmc, dim=1)  # [B,n]
        score = fmc #mf + fmc
        #score = torch.squeeze(score)
        return score
        
    def fit(self, train_loader, valid_loader=None):
        self.to(self.device)
        self.logger.info('Start training...')
        last_loss = 0.0
        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = []
            current_loss = 0.0
            for _, (seq, target, lens, _) in enumerate(train_loader):
                seq = seq.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                logit = self.forward(seq, lens)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                total_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            current_loss = np.mean(total_loss)
            delta_loss = current_loss - last_loss
            if abs(delta_loss) < 1e-5:
                self.logger.info(f'early stop at epoch {epoch}')
                break
            last_loss = current_loss


            s = ''
            if valid_loader:
                self.eval()
                val_loss = []
                with torch.no_grad():
                    for _, (seq, target, lens) in enumerate(valid_loader):
                        seq = seq.to(self.device)
                        target = target.to(self.device)
                        logit = self.forward(seq, lens)
                        logit_sampled = logit[:, target.view(-1)]
                        loss = self.loss_func(logit_sampled)
                        val_loss.append(loss.item())
                s = f'\tValidation Loss: {np.mean(val_loss):3f}'

            self.logger.info(f'training epoch: {epoch}\tTrain Loss: {np.mean(total_loss):.3f}' + s)
            
    def predict(self, test_loader, k=15):
        self.eval()
        preds, last_item = torch.tensor([]), torch.tensor([])
        for _, (seq, target_item, lens, candidate_set) in enumerate(test_loader):
            scores = self.forward(seq.to(self.device), lens)
            # scores = torch.gather(scores, dim=1, index=candidate_set.to(self.device))
            # rank_list = (torch.argsort(scores[:,1:], descending=True) + 1)[:,:k]  # why +1: +1 to represent the actual code of items
            scores = torch.gather(scores, 1, candidate_set.to(self.device)).topk(k)[1]
            rank_list = torch.gather(candidate_set.to(self.device), 1, scores)
            preds = torch.cat((preds, rank_list.cpu()), 0)
            last_item = torch.cat((last_item, target_item), 0)

        return preds, last_item

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        # differences between the item scores
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        # final loss
        loss = -torch.mean(F.logsigmoid(diff))
        return loss