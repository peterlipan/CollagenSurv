import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ModelOutputs


class Pooler(nn.Module):
    def __init__(self, d_in, d_model, n_classes, p_method='mean', activation='tanh'):
        super(Pooler, self).__init__()
        self.p_method = p_method
        self.d_model = d_model
        self.linear = nn.Linear(d_in, d_model)
        self.classifier = nn.Linear(d_model, n_classes)

        if p_method == 'mean':
            self.pooler = nn.AdaptiveAvgPool1d(1)
        elif p_method == 'max':
            self.pooler = nn.AdaptiveMaxPool1d(1)
        else:
            raise NotImplementedError
        
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise NotImplementedError
        
    
    def forward(self, x):
        # x: [B, N, D], B: batch size, N: number of patches, D: feature dimension
        x = self.pooler(x.transpose(1, 2)).squeeze(-1) # [B, D]
        x = self.linear(x)
        features = self.act(x)
        logits = self.classifier(features)
        y_hat = torch.argmax(logits, dim=1)
        y_prob = F.softmax(logits, dim=1)

        return ModelOutputs(features=features, logits=logits, y_hat=y_hat, y_prob=y_prob)
