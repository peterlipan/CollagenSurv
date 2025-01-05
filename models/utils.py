import torch
import torch.nn as nn


class ModleOutputs:
    def __init__(self, features=None, logits=None, y_prob=None, y_hat=None):
        self.dict = {'features': features, 'logits': logits, 'y_prob': y_prob, 'y_hat': y_hat}
    
    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
    
    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return str(self.dict)


def CreateModel(args):
    if args.backbone.lower() == 'transmil':
        from .TransMIL import TransMIL
        model = TransMIL(args.d_in, args.d_model, args.n_classes)
    elif args.backbone.lower() == 'abmil':
        from .ABMIL import DAttention
        model = DAttention(args.d_in, args.n_classes, dropout=args.dropout, act=args.act)
    elif args.backbone.lower() == 'meanpool':
        from .Pooler import Pooler
        model = Pooler(args.d_in, args.d_model, args.n_classes, p_method='mean', activation=args.act)
    elif args.backbone.lower() == 'maxpool':
        from .Pooler import Pooler
        model = Pooler(args.d_in, args.d_model, args.n_classes, p_method='max', activation=args.act)
    else:
        raise NotImplementedError
