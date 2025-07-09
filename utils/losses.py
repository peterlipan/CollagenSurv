import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pycox.models.loss import CoxPHLoss, DeepHitSingleLoss


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
    def __init__(self, alpha=0.15, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.cph = CoxPHLoss()
        self.eps = eps

    def forward(self, outputs, data): 
        loss = self.cph(outputs.risk.add(self.eps).log(), data['duration'], data['event']) + ce_loss(outputs.hazards, outputs.surv, data['label'], data['c'], alpha=self.alpha)
        return loss
    

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
        risk_clamped = outputs.risk.clamp(min=self.eps) # for numerical stability
        return self.cph(risk_clamped.log(), data['duration'], data['event'])


class DeepHitsurvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dhl = DeepHitSingleLoss(alpha=0.5, sigma=0.5)
    
    @staticmethod
    def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
        n = len(idx_durations)
        for i in range(n):
            dur_i = idx_durations[i]
            ev_i = events[i]
            if ev_i == 0:
                continue
            for j in range(n):
                dur_j = idx_durations[j]
                ev_j = events[j]
                if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                    mat[i, j] = 1
        return mat

    def pair_rank_mat(self, idx_durations, events, dtype='float32'):
        """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
        So it takes value 1 if we observe that i has an event before j and zero otherwise.
        
        Arguments:
            idx_durations {np.array} -- Array with durations.
            events {np.array} -- Array with event indicators.
        
        Keyword Arguments:
            dtype {str} -- dtype of array (default: {'float32'})
        
        Returns:
            np.array -- n x n matrix indicating if i has an observerd event before j.
        """
        idx_durations = idx_durations.reshape(-1)
        events = events.reshape(-1)
        n = len(idx_durations)
        mat = np.zeros((n, n), dtype=dtype)
        mat = self._pair_rank_mat(mat, idx_durations, events, dtype)
        return mat

    def forward(self, outputs, data):
        # directly predict the risk factors
        rank_mat = self.pair_rank_mat(data['duration'].cpu().numpy(), data['event'].cpu().numpy())
        rank_mat = torch.tensor(rank_mat, dtype=outputs.hazards.dtype, device=outputs.hazards.device)
        return self.dhl(outputs.hazards, data['label'], data['event'], rank_mat)
