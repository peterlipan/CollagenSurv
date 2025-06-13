import torch
import torchvision
import torch.nn as nn


class ModelOutputs:
    def __init__(self, features=None, logits=None, y_prob=None, y_hat=None, hazards=None, surv=None, **kwargs):
        self.dict = {'features': features, 'logits': logits, 'y_prob': y_prob, 'y_hat': y_hat, 'hazards': hazards, 'surv': surv}
        self.dict.update(kwargs)
    
    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
    
    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return str(self.dict)

    def __getattr__(self, key):
        return self.dict[key]


class CreateModel(nn.Module):
    def __init__(self, args):
        backbone, pretrained, n_classes = args.backbone, args.pretrained, args.n_classes
        super(CreateModel, self).__init__()
        models = ['resnet18', 'resnet50', 'efficientnet_v2_s']
        assert backbone in models
        model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.n_classes = n_classes
        
        if backbone.startswith('resnet'):
            self.n_features = model.fc.in_features
            model.fc = nn.Identity()

        elif backbone.startswith('densenet'):
            self.n_features = model.classifier.in_features
            model.classifier = nn.Identity()
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

        elif backbone.startswith('efficient'):
            self.n_features = model.classifier[1].in_features
            model.classifier[1] = nn.Identity()

        head = nn.Linear(self.n_features, n_classes, bias=True)

        self.encoder = model
        self.head = head

    def forward(self, x):

        features = self.encoder(x['image'])
        logits = self.head(features)

        y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        return ModelOutputs(features=features, logits=logits, hazards=hazards, surv=surv, y_hat=y_hat)
