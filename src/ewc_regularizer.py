from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from src.config import Config as c


class EWC(object):
    def __init__(self, model, dataloader, device, scaled_hypo):

        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.scaled_hypo = scaled_hypo
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        criterion_reg, criterion_clf = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([c.clf_hypo_w]).to(c.device))
        
        precision_matrices = {n: torch.zeros(p.size()).to(self.device) for n, p in self.params.items()}

        self.model.train()
        # with torch.no_grad():
        for batch_X, batch_y in self.dataloader:
            batch_X, batch_y = batch_X.to(self.device), batch_y[:,-1,0].to(self.device)
            batch_y_binary = (batch_y < self.scaled_hypo)*1.0

            self.model.zero_grad()
            output_reg, output_clf = self.model(batch_X)
            reg_loss = criterion_reg(output_reg.reshape(-1,1), batch_y.reshape(-1,1)) * c.reg_w
            clf_loss = criterion_clf(output_clf.reshape(-1,1), batch_y_binary.reshape(-1,1))
            loss = reg_loss + clf_loss
            
            loss.backward()
            
            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)
        
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices
        
    # penalty = diag * squared difference of [After 1st Task] & [current]    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss