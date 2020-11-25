import torch
from torchvision import models

from .projection_head import MLPHead


class ResNetSimCLR(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNetSimCLR, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        elif kwargs['name'] == 'resnet50_2':
            resnet = models.wide_resnet50_2(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(in_channels=resnet.fc.in_features, **kwargs['mlp_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return h, self.projection(h)
