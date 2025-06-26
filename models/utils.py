import torch
import torchvision
import torch.nn as nn
from .MyViT import ViTModel, ViTConfig


class Transformer(nn.Module):
    def __init__(self, image_size, num_classes, pretrained="", patch_size=4):
        super(Transformer, self).__init__()
        config = ViTConfig().from_pretrained(pretrained) if pretrained else ViTConfig()
        config.num_labels = num_classes
        config.image_size = image_size
        config.patch_size = patch_size
        
        self.config = config
        self.num_classes = num_classes
        self.hidden_size = config.hidden_size
        
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True, use_cls_token=False)
        if pretrained:
            self.vit = ViTModel.from_pretrained(pretrained, config=config, add_pooling_layer=False, use_mask_token=False, use_cls_token=False, ignore_mismatched_sizes=True)
        # replace the official ViT 'pooler' as the real pooling layer
        self.pooler = nn.AdaptiveAvgPool1d(1)

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, x, token_mask=None):
        return_dict = self.config.use_return_dict
        outputs = self.vit(x, bool_masked_pos=token_mask, return_dict=return_dict)
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output.transpose(1, 2)[:, :, 1:])
        pooled_output = torch.flatten(pooled_output, 1)
        return pooled_output


class CustomCNNBackbone(nn.Module):
    def __init__(self, out_dim=512):
        super(CustomCNNBackbone, self).__init__()
        self.features = nn.Sequential(
            # Input: [B, 3, 1200, 1200]
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # -> [B, 32, 600, 600]
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # -> [B, 32, 300, 300]

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> [B, 64, 150, 150]
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> [B, 128, 75, 75]
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # -> [B, 256, 38, 38]
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # -> [B, 512, 19, 19]
            nn.BatchNorm2d(512),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 512, 1, 1]
        )
        self.out_dim = out_dim
        self.projector = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.projector(x)  # Final feature vector


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
        

        self.n_classes = n_classes
        self.task = args.task
        # model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        # if backbone.startswith('resnet'):
        #     self.n_features = model.fc.in_features
        #     model.fc = nn.Identity()

        # elif backbone.startswith('densenet'):
        #     self.n_features = model.classifier.in_features
        #     model.classifier = nn.Identity()
        #     for m in model.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.constant_(m.bias, 0)

        # elif backbone.startswith('efficient'):
        #     self.n_features = model.classifier[1].in_features
        #     model.classifier[1] = nn.Identity()

        # head = nn.Linear(self.n_features, n_classes, bias=True)

        self.encoder = Transformer(
            image_size=args.image_size,
            num_classes=n_classes,
            pretrained=pretrained,
            patch_size=args.patch_size
        )
        n_features = self.encoder.hidden_size
        self.head = nn.Linear(n_features, n_classes)

    def feature_forward(self, x):
        features = self.encoder(x['image'])
        logits = self.head(features)
        return features, logits

    def surv_forward(self, x):

        features, logits = self.feature_forward(x)

        y_hat = torch.argmax(logits, dim=1)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        return ModelOutputs(features=features, logits=logits, hazards=hazards, surv=surv, y_hat=y_hat)
    
    def cls_forward(self, x):
        features, logits = self.feature_forward(x)
        y_prob = torch.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)
        return ModelOutputs(features=features, logits=logits, y_prob=y_prob, y_hat=y_hat)
    
    def forward(self, x):
        if self.task == 'survival':
            return self.surv_forward(x)
        elif self.task in ['grade', 'er', 'pr', 'her2', 'node_status']:
            return self.cls_forward(x)
        else:
            raise ValueError(f"Unsupported task: {self.task}")
