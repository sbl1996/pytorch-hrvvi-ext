import torch
import torch.nn as nn
import timm


class SpinalNet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(1280, 22, 1)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.head(x)
        return x

backbone = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
backbone.blocks[3][0].conv_dw.stride = (1, 1)
backbone.blocks[5][0].conv_dw.stride = (1, 1)
model = SpinalNet(backbone)
x = torch.randn(2, 3, 224, 224)
print(model(x).shape)