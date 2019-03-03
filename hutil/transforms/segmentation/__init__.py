import torch
import numpy as np

import torchvision.transforms.functional as TF
from hutil.transforms import JointTransform


class SameTransform(JointTransform):

    def __init__(self, t):
        super().__init__()
        self.t = t

    def __call__(self, img, seg_img):
        return self.t(img), self.t(seg_img)


class ToTensor(JointTransform):
    """Convert the input ``PIL Image`` to tensor and the target segmentation image to labels.

    For the segmentation labels, 0 represents background and `num_classes` + 1 represents border.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, img, seg_img):
        input = TF.to_tensor(input)
        target = np.array(seg_img)
        target[target == 255] = self.num_classes + 1
        target = torch.from_numpy(target).long()
        return self.transform(input, target)
