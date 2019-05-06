import torch
import torch.nn as nn


def _seq(x):
    if torch.is_tensor(x):
        return (x,)
    else:
        return x


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
        loc_p, cls_p = self.head(*_seq(ps))
        return loc_p, cls_p

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        dets = self._inference(*_seq(preds))
        self.train()
        return dets
