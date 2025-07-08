import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pycox.models.loss import CoxPHLoss


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
   
    S_padded = torch.cat([torch.ones_like(c), S], 1) 

    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1, 1 means censored, 0 means uncensored
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y)+eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(nn.Module):
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, data): 
        return ce_loss(outputs.hazards, outputs.surv, data['label'], data['c'], alpha=self.alpha)
    

class CrossEntropyClsLoss(nn.CrossEntropyLoss):
    def forward(self, outputs, data):
        return F.cross_entropy(
            outputs['logits'],
            data['label'],
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )



class NLLSurvLoss(nn.Module):
    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, data):

        return nll_loss(outputs.hazards, outputs.surv, data['label'], data['c'], alpha=self.alpha)


class CoxSurvLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.cph = CoxPHLoss()
        self.eps = eps
    def forward(self, outputs, data):
        # directly predict the risk factors
        return self.cph(outputs.risk.add(self.eps).log(), data['duration'], data['event'])
