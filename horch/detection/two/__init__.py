import random
import warnings

from horch.nn.loss import loc_kl_loss
from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.common import select, sample, _concat, one_hot, expand_last_dim
from horch.detection.one import MultiBoxLoss, AnchorBasedInference, match_anchors, flatten, MultiBoxLossOnline
from horch.detection.bbox import BBox
from horch.detection.nms import nms, soft_nms_cpu, softer_nms_cpu


def sample_rois(loc_t, cls_t, n_samples=64, neg_pos_ratio=3):
    pos = cls_t != 0
    n_pos = int(n_samples / (neg_pos_ratio + 1))
    n_neg = n_samples - n_pos
    pos_indices = sample(torch.nonzero(pos).squeeze(1), n_pos)
    neg_indices = sample(torch.nonzero(~pos).squeeze(1), n_neg)
    loc_t = loc_t[pos_indices]
    indices = torch.cat([pos_indices, neg_indices], dim=0)
    cls_t = cls_t[indices]
    return loc_t, cls_t, indices


@curry
def inference_rois(loc_p, cls_p, anchors, iou_threshold=0.5, topk=100, conf_strategy='softmax'):
    if conf_strategy == 'softmax':
        scores = torch.softmax(cls_p, dim=1)
    else:
        scores = torch.sigmoid_(cls_p)
    scores = torch.max(scores[..., 1:], dim=-1)[0]

    loc_p[..., :2].mul_(anchors[:, 2:]).add_(anchors[:, :2])
    loc_p[..., 2:].exp_().mul_(anchors[:, 2:])

    bboxes = BBox.convert(
        loc_p, format=BBox.XYWH, to=BBox.LTRB, inplace=True)

    rois = []
    for i in range(len(loc_p)):
        ibboxes = bboxes[i]
        iscores = scores[i]
        indices = nms(ibboxes, iscores, iou_threshold)
        ibboxes = ibboxes[indices]
        iscores = iscores[indices]
        if len(indices) > topk:
            indices = iscores.topk(topk)[1]
            ibboxes = ibboxes[indices]
        else:
            ibboxes = sample(ibboxes, topk)
        batch_idx = ibboxes.new_full((topk, 1), i)
        rois.append(torch.cat([batch_idx, ibboxes], dim=-1))
    rois = torch.stack(rois, dim=0)
    return rois


class InferenceRoIs:

    def __init__(self, anchors, conf_threshold=0.01,
                 iou_threshold=0.5, topk=100, conf_strategy='softmax'):
        self.anchors = flatten(anchors)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.topk = topk
        assert conf_strategy in [
            'softmax', 'sigmoid'], "conf_strategy must be softmax or sigmoid"
        self.conf_strategy = conf_strategy

    def __call__(self, loc_p, cls_p):
        rois = inference_rois(
            loc_p, cls_p, self.anchors, self.iou_threshold, self.topk, self.conf_strategy)
        return rois


class MatchRoIs:
    def __init__(self, pos_thresh=0.5, n_samples=None, neg_pos_ratio=3):
        super().__init__()
        self.pos_thresh = pos_thresh
        self.n_samples = n_samples
        self.neg_pos_ratio = neg_pos_ratio

    def __call__(self, rois, box_lists):
        is_cpu = rois.device.type == 'cpu'
        roi_corners = rois[..., 1:]
        roi_centers = BBox.convert(roi_corners, BBox.LTRB, BBox.XYWH)

        bboxes = [
            BBox.convert(rois.new_tensor([b['bbox'] for b in boxes]), BBox.LTWH, BBox.XYWH)
            for boxes in box_lists
        ]
        labels = [
            rois.new_tensor([b['category_id'] for b in boxes], dtype=torch.long)
            for boxes in box_lists
        ]

        loc_targets = []
        cls_targets = []
        sampled_rois = []
        for i in range(len(rois)):
            loc_t, cls_t, _ = match_anchors(
                bboxes[i], labels[i], roi_centers[i], roi_corners[i],
                self.pos_thresh, neg_thresh=None, cpu=is_cpu)
            loc_t, cls_t, indices = sample_rois(loc_t, cls_t, self.n_samples, self.neg_pos_ratio)
            loc_targets.append(loc_t)
            cls_targets.append(cls_t)
            sampled_rois.append(rois[i][indices])
        loc_t = torch.cat(loc_targets, dim=0)
        cls_t = torch.cat(cls_targets, dim=0)
        rois = torch.cat(sampled_rois, dim=0)

        return loc_t, cls_t, rois


class RCNNLoss(nn.Module):

    def __init__(self, matcher, p=0.01, rpn_cls_loss='ce'):
        super().__init__()
        self.rpn_loss = MultiBoxLossOnline(matcher, neg_pos_ratio=1, p=p, cls_loss=rpn_cls_loss, prefix='RPN')
        self.rcnn_loss = MultiBoxLoss(p=p, prefix='RCNN')

    @property
    def p(self):
        return self.rcnn_loss.p

    @p.setter
    def p(self, new_p):
        self.rpn_loss.p = new_p
        self.rcnn_loss.p = new_p

    def forward(self, rois, rpn_loc_p, rpn_cls_p, loc_p, cls_p, loc_t, cls_t):
        rpn_loc_p = rpn_loc_p.view(-1, 4)
        rpn_cls_p = rpn_cls_p.view(-1, rpn_cls_p.size(-1))
        rpn_loss = self.rpn_loss(rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore)

        # num_classes = cls_p.size(-1) - 1
        # loc_p = expand_last_dim(loc_p, num_classes, 4)
        #
        # pos = cls_t != 0
        # loc_p = select(loc_p[pos], 1, cls_t[pos] - 1)

        rcnn_loss = self.rcnn_loss(loc_p, cls_p, loc_t, cls_t)
        loss = rpn_loss + rcnn_loss
        return loss


@curry
def roi_based_inference(
        rois, loc_p, cls_p, conf_threshold=0.01,
        iou_threshold=0.5, topk=100, nms_method='soft'):

    scores, labels = torch.softmax(cls_p, dim=1)[:, 1:].max(dim=1)
    # num_classes = cls_p.size(1) - 1
    # loc_p = expand_last_dim(loc_p, num_classes, 4)
    # loc_p = select(loc_p, 1, labels)
    bboxes = loc_p

    if conf_threshold:
        pos = scores > conf_threshold
        bboxes = bboxes[pos]
        rois = rois[pos]
        scores = scores[pos]
        labels = labels[pos]

    bboxes[..., :2].mul_(rois[:, 2:]).add_(rois[:, :2])
    bboxes[..., 2:].exp_().mul_(rois[:, 2:])

    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
    scores = scores.cpu()

    if nms_method == 'soft':
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk)
    else:
        indices = nms(bboxes, scores, iou_threshold)
        if len(indices) > topk:
            indices = indices[scores[indices].topk(topk)[1]]
        else:
            warnings.warn("Only %d RoIs left after nms rather than top %d" % (len(scores), topk))
    bboxes = BBox.convert(
        bboxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)

    dets = []
    for i, ind in enumerate(indices):
        det = {
            'image_id': -1,
            'category_id': labels[ind].item() + 1,
            'bbox': bboxes[ind].tolist(),
            'score': scores[ind].item(),
        }
        dets.append(det)
    return dets


class RoIBasedInference:

    def __init__(self, iou_threshold=0.5, conf_threshold=0.01, topk=100, nms='soft'):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
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
                self.iou_threshold, self.conf_threshold, self.topk, self.nms)
            image_dets.append(dets)
        return image_dets
