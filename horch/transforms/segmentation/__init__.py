import torch
import numpy as np

import torchvision.transforms.functional as TF
from horch.transforms import JointTransform


class SameTransform(JointTransform):

    def __init__(self, t):
        super().__init__()
        self.t = t

    def __call__(self, img, seg):
        return self.t(img), self.t(seg)


class ToTensor(JointTransform):
    """Convert the input ``PIL Image`` to tensor and the target segmentation image to labels.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, img, seg):
        input = TF.to_tensor(img)
        target = np.array(seg)
        target = torch.from_numpy(target).long()
        return input, target
