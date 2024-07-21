import timm
import torch
import torch.nn as nn


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained_path=None, features_only=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,  
            features_only=features_only,
            in_chans=in_c,
            num_classes=n_classes,
            global_pool='avg'
        )
        if pretrained_path:
            state_dict = torch.load(pretrained_path)
            # Adapt the model to new input channels and output classes
            self.model.classifier = nn.Linear(self.model.classifier.in_features, n_classes)
            self.model.features.conv0 = nn.Conv2d(in_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            state_dict['classifier.weight'] = self.model.classifier.weight
            state_dict['classifier.bias'] = self.model.classifier.bias
            state_dict['features.conv0.weight'] = self.model.features.conv0.weight
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        y = self.model(x)
        return y


# class RSNA24Model(nn.Module):
#     def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
#         super().__init__()
#         self.model = timm.create_model(
#                                     model_name,
#                                     pretrained=pretrained, 
#                                     features_only=features_only,
#                                     in_chans=in_c,
#                                     num_classes=n_classes,
#                                     global_pool='avg'
#                                     )
    
#     def forward(self, x):
#         y = self.model(x)
#         return y
    