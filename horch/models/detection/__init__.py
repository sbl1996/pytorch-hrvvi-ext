import torch
import torch.nn as nn

from horch.common import _tuple


class OneStageDetector(nn.Module):
    def __init__(self, backbone, fpn, head, inference):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self._inference = inference

    def forward(self, x):
        cs = self.backbone(x)
        ps = self.fpn(*cs)
        return self.head(*_tuple(ps))

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        dets = self._inference(*_tuple(preds))
        self.train()
        return dets
