import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NewDropout(nn.Module):
    def __init__(self, p=0.5):
        super(NewDropout, self).__init__()
        self.p = p
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0 - p)
        else:
            self.multiplier_ = 0.0

    def forward(self, input):
        if not self.training:
            return input
        
        selected_ = torch.Tensor(input.shape).uniform_(0, 1) > self.p
        selected_ = Variable(selected_.type_as(input), requires_grad=False)
            
        return torch.mul(selected_, input) * self.multiplier_

class MC_MTL_SGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_rate=0.5):
        super(MC_MTL_SGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
            
        self.gru = nn.ModuleList(
            nn.GRU(input_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True) 
            for i in range(n_layers))
        self.dropout = NewDropout(dropout_rate)
        self.fc_reg = nn.Linear(hidden_dim, output_dim//2)
        self.fc_clf = nn.Linear(hidden_dim, output_dim//2)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.gru:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
        
        nn.init.xavier_uniform_(self.fc_reg.weight)
        nn.init.constant_(self.fc_reg.bias, 0)
        nn.init.xavier_uniform_(self.fc_clf.weight)
        nn.init.constant_(self.fc_clf.bias, 0)

    def forward(self, x):
        h = None
        for i, layer in enumerate(self.gru):
            x, h = layer(x, h)
            x = self.dropout(x)  # Apply MyDropout to all layers
        
        y_pred_reg = self.fc_reg(x[:, -1])
        y_pred_clf = self.fc_clf(x[:, -1])
        return y_pred_reg, y_pred_clf