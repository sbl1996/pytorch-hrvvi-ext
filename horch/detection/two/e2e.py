from toolz import curry

from torch.utils.data.dataloader import default_collate

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.utils import _concat
from horch.detection.one import MultiBoxLoss, coords_to_target, target_to_coords, match_anchors_flat, AnchorBasedInference
from horch.detection.bbox import BBox
from horch.detection.iou import iou_mn
from horch.detection.nms import nms_cpu, soft_nms_cpu


def match_rois(anns, rois_ltrb, rois_xywh, pos_thresh=0.5, n_samples=64, pos_neg_ratio=1/3):
    num_rois = len(rois_ltrb)
    loc_t = rois_ltrb.new_zeros(num_rois, 4)
    cls_t = loc_t.new_zeros(num_rois, dtype=torch.long)

    if len(anns) == 0:
        return loc_t, cls_t

    bboxes = loc_t.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = loc_t.new_tensor([ann['category_id'] for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, rois_ltrb)

    max_ious, indices = ious.max(dim=1)
    loc_t[indices] = coords_to_target(bboxes, rois_xywh[indices])
    cls_t[indices] = labels

    pos = ious > pos_thresh
    for ipos, bbox, label in zip(pos, bboxes, labels):
        loc_t[ipos] = coords_to_target(bbox, rois_xywh[ipos])
        cls_t[ipos] = label

    pos = cls_t != 0
    n_pos = int(n_samples * pos_neg_ratio / (pos_neg_ratio + 1))
    n_neg = n_samples - n_pos
    pos_indices = sample(torch.nonzero(pos).squeeze(1), n_pos)
    neg_indices = sample(torch.nonzero(~pos).squeeze(1), n_neg)
    loc_t = loc_t[pos_indices]
    indices = torch.cat([pos_indices, neg_indices], dim=0)
    cls_t = cls_t[indices]
    return loc_t, cls_t, indices


def sample(t, n):
    if len(t) >= n:
        indices = torch.randperm(len(t), device=t.device)[:n]
    else:
        indices = torch.randint(len(t), size=(n,), device=t.device)
    return t[indices]


class MatchRoIs:
    def __init__(self, pos_thresh=0.5, n_samples=None, pos_neg_ratio=1/3):
        super().__init__()
        self.pos_thresh = pos_thresh
        self.n_samples = n_samples
        self.pos_neg_ratio = pos_neg_ratio

    def __call__(self, rois, image_gts):
        batch_size, num_rois = rois.size()[:2]
        rois_ltrb = rois[..., 1:]
        rois_xywh = BBox.convert(rois_ltrb, format=BBox.LTRB, to=BBox.XYWH)
        loc_targets = []
        cls_targets = []
        sampled_rois = []
        for i in range(batch_size):
            loc_t, cls_t, indices = match_rois(image_gts[i], rois_ltrb[i], rois_xywh[i], self.pos_thresh, self.n_samples, self.pos_neg_ratio)
            loc_targets.append(loc_t)
            cls_targets.append(cls_t)
            sampled_rois.append(rois[i][indices])
        loc_t = torch.cat(loc_targets, dim=0)
        cls_t = torch.cat(cls_targets, dim=0)
        rois = torch.cat(sampled_rois, dim=0)

        return loc_t, cls_t, rois


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return _concat(xs, dim=0)


class MatchAnchors:
    r"""

    Parameters
    ----------
    anchors : torch.Tensor or List[torch.Tensor]
        List of anchor boxes of shape `(lx, ly, #anchors, 4)`.
    pos_thresh : float
        IOU threshold of positive anchors.
    neg_thresh : float
        If provided, only non-positive anchors whose ious with all ground truth boxes are
        lower than neg_thresh will be considered negative. Other non-positive anchors will be ignored.
    get_label : function
        Function to extract label from annotations.
    """

    def __init__(self, anchors, pos_thresh=0.5, neg_thresh=None, debug=False):
        self.anchors_xywh = flatten(anchors)
        self.anchors_ltrb = BBox.convert(self.anchors_xywh, BBox.XYWH, BBox.LTRB)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.get_label = lambda x: 1
        self.debug = debug

    def __call__(self, img, anns):
        target = match_anchors_flat(
            anns, self.anchors_xywh, self.anchors_ltrb,
            self.pos_thresh, self.neg_thresh, self.get_label, self.debug)
        return img, target


class RCNNLoss(nn.Module):

    def __init__(self, p=0.01):
        super().__init__()
        self.rpn_loss = MultiBoxLoss(p=p, criterion='focal', prefix='RPN')
        self.rcnn_loss = MultiBoxLoss(p=p, prefix='RCNN')

    @property
    def p(self):
        return self.rcnn_loss.p

    @p.setter
    def p(self, new_p):
        self.rpn_loss.p = new_p
        self.rcnn_loss.p = new_p

    def forward(self, loc_p, cls_p, loc_t, cls_t, rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore):
        rpn_loss = self.rpn_loss(rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore)
        rcnn_loss = self.rcnn_loss(loc_p, cls_p, loc_t, cls_t)
        loss = rpn_loss + rcnn_loss
        return loss


@curry
def roi_based_inference(
        rois, loc_p, cls_p, conf_threshold=0.01,
        iou_threshold=0.5, topk=100, nms='soft_nms'):
    dets = []
    bboxes = loc_p
    logits = cls_p
    scores = torch.softmax(logits, dim=1)[:, 1:]
    scores, labels = torch.max(scores, dim=1)

    if conf_threshold > 0:
        pos = scores > conf_threshold
        scores = scores[pos]
        labels = labels[pos]
        bboxes = bboxes[pos]
        rois = rois[pos]

    bboxes = target_to_coords(bboxes, rois)

    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
    scores = scores.cpu()

    if nms == 'nms':
        indices = nms_cpu(bboxes, scores, iou_threshold)
        scores = scores[indices]
        labels = labels[indices]
        bboxes = bboxes[indices]
        if scores.size(0) > topk:
            indices = scores.topk(topk)[1]
        else:
            indices = range(scores.size(0))
    else:
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk, conf_threshold=conf_threshold)
    bboxes = BBox.convert(
        bboxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)
    for ind in indices:
        det = {
            'image_id': -1,
            'category_id': labels[ind].item() + 1,
            'bbox': bboxes[ind].tolist(),
            'score': scores[ind].item(),
        }
        dets.append(det)
    return dets


class RoIBasedInference:

    def __init__(self, conf_threshold=0.01, iou_threshold=0.5, topk=100, nms='soft_nms'):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        self.nms = nms

    def __call__(self, rois, loc_p, cls_p):
        image_dets = []
        batch_size, num_rois = rois.size()[:2]
        rois = BBox.convert(rois, BBox.LTRB, BBox.XYWH, inplace=True)
        loc_p = loc_p.view(batch_size, num_rois, -1)
        cls_p = cls_p.view(batch_size, num_rois, -1)
        for i in range(batch_size):
            dets = roi_based_inference(
                rois[i], loc_p[i], cls_p[i],
                self.conf_threshold, self.iou_threshold, self.topk, self.nms)
            image_dets.append(dets)
        return image_dets


@curry
def input_only_collate(batch):
    input, target = zip(*batch)
    if any([torch.is_tensor(t) for t in target[0]]):
        target = [default_collate(t) if torch.is_tensor(t[0]) else t for t in zip(*target)]
    return [default_collate(input), target], []