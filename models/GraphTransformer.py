# CLAIM: The following and relevant disgusting code is from the original authors of the paper.
# I tried to clean it up, but it's still a mess. I'm sorry.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ViT import VisionTransformer
from .GCN import GCNBlock
from .utils import ModelOutputs
from torch_geometric.nn import dense_mincut_pool


class GraphTransformer(nn.Module): # TODO: Implement multi-task learning
    def __init__(self, d_in, n_classes, surv_classes=4, task='cls'):
        super().__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100

        self.transformer = VisionTransformer(num_classes=n_classes, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.criterion = nn.CrossEntropyLoss()

        self.conv1 = GCNBlock(d_in, self.embed_dim)  # 64->128
        self.pool1 = nn.Linear(self.embed_dim, self.node_cluster_num)  # 100-> 20

    def cls_forward(self, data):

        x, adj = data['x'].float(), data['adj'].float()
        bs = x.size(0)

        x = self.conv1(x, adj)
        s = self.pool1(x)

        x, adj, *_ = dense_mincut_pool(x, adj, s)
        cls_token = self.cls_token.repeat(bs, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        logits, _ = self.transformer(x)

        y_hat = torch.argmax(logits, dim=1)
        y_prob = F.softmax(logits, dim=1)

        return ModelOutputs(features=x, logits=logits, y_hat=y_hat, y_prob=y_prob)
    
    def surv_forward(self, data):
        x, adj = data['x'].float(), data['adj'].float()
        bs = x.size(0)

        x = self.conv1(x, adj)
        s = self.pool1(x)

        x, adj, *_ = dense_mincut_pool(x, adj, s)
        cls_token = self.cls_token.repeat(bs, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        logits, _ = self.transformer(x)

        y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        return ModelOutputs(features=x, logits=logits, y_hat=y_hat, hazards=hazards, surv=surv)
    
    def forward(self, data):
        if 'cls' in self.task:
            return self.cls_forward(data)
        elif 'surv' in self.task:
            return self.surv_forward(data)
        else:
            raise NotImplementedError
