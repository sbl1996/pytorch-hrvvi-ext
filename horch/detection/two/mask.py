import warnings
import random

from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.ops import sample, _concat, expand_last_dim, select
from horch.detection.one import MultiBoxLoss
from horch.detection.bbox import BBox
from horch.detection.iou import iou_mn
from horch.detection.nms import nms, soft_nms_cpu
from horch.detection.two import MatchAnchors, coords_to_target2, coords_to_target


def match_rois(anns, rois, pos_thresh=0.5, mask_size=(14, 14), n_samples=64, pos_neg_ratio=1 / 3):
    rois_xywh = BBox.convert(rois, BBox.LTRB, BBox.XYWH)
    num_anns = len(anns)
    num_rois = len(rois)
    loc_t = rois.new_zeros(num_rois, 4)
    cls_t = loc_t.new_zeros(num_rois, dtype=torch.long)

    if num_anns == 0:
        return loc_t, cls_t

    bboxes = loc_t.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = loc_t.new_tensor([ann['category_id'] for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, rois)

    ann_indices = torch.zeros(num_rois, dtype=torch.long)
    max_ious, indices = ious.max(dim=1)
    loc_t[indices] = coords_to_target(bboxes, rois_xywh[indices])
    cls_t[indices] = labels
    ann_indices[indices] = torch.arange(num_anns)

    pos = ious > pos_thresh
    for ann_id, ipos, bbox, label in zip(range(num_rois), pos, bboxes, labels):
        loc_t[ipos] = coords_to_target(bbox, rois_xywh[ipos])
        cls_t[ipos] = label
        ann_indices[ipos] = ann_id

    pos = cls_t != 0
    n_pos = int(n_samples * pos_neg_ratio / (pos_neg_ratio + 1))
    n_neg = n_samples - n_pos
    pos_indices = sample(torch.nonzero(pos).squeeze(1), n_pos)
    neg_indices = sample(torch.nonzero(~pos).squeeze(1), n_neg)
    loc_t = loc_t[pos_indices]
    indices = torch.cat([pos_indices, neg_indices], dim=0)
    cls_t = cls_t[indices]

    mask_t = loc_t.new_zeros(n_pos, *mask_size)
    for i in range(n_pos):
        ind = pos_indices[i]
        mask = anns[ann_indices[ind]]['segmentation']
        height, width = mask.shape
        l, t, r, b = rois[ind]
        l = max(0, int(l * width))
        t = max(0, int(t * height))
        r = int(r * width)
        b = int(b * height)
        m = mask[t:b, l:r].float()
        m = m.view(1, 1, *m.size())
        m = F.interpolate(m, size=mask_size).squeeze()
        mask_t[i] = m
    return loc_t, cls_t, mask_t, indices


def match_rois2(anns, rois, pos_thresh=0.5, mask_size=(14, 14), n_samples=64, pos_neg_ratio=1 / 3):
    num_rois = len(rois)
    if len(anns) == 0:
        loc_t = rois.new_zeros(num_rois, 4)
        cls_t = loc_t.new_zeros(num_rois, dtype=torch.long)
        return loc_t, cls_t

    rois_xywh = BBox.convert(rois, BBox.LTRB, BBox.XYWH)

    bboxes = rois.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = rois.new_tensor([ann['category_id'] for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, rois)

    pos = ious > pos_thresh
    cls_t, ann_indices = (pos.long() * labels[:, None]).max(dim=0)
    loc_t_all = coords_to_target2(bboxes, rois_xywh)
    loc_t = select(loc_t_all, 0, ann_indices)

    max_ious, max_indices = ious.max(dim=1)
    loc_t[max_indices] = select(loc_t_all, 1, max_indices)
    cls_t[max_indices] = labels

    ann_indices[max_indices] = torch.arange(len(anns), device=rois.device)

    pos = cls_t != 0
    n_pos = int(n_samples * pos_neg_ratio / (pos_neg_ratio + 1))
    n_neg = n_samples - n_pos
    pos_indices = sample(torch.nonzero(pos).squeeze(1), n_pos)
    neg_indices = sample(torch.nonzero(~pos).squeeze(1), n_neg)
    loc_t = loc_t[pos_indices]
    indices = torch.cat([pos_indices, neg_indices], dim=0)
    cls_t = cls_t[indices]

    mask_t = loc_t.new_zeros(n_pos, *mask_size)
    for i in range(n_pos):
        ind = pos_indices[i]
        mask = anns[ann_indices[ind]]['segmentation']
        height, width = mask.shape
        l, t, r, b = rois[ind]
        l = max(0, int(l * width))
        t = max(0, int(t * height))
        r = int(r * width)
        b = int(b * height)
        m = mask[t:b, l:r].float()
        m = m.view(1, 1, *m.size())
        m = F.interpolate(m, size=mask_size).squeeze()
        mask_t[i] = m
    return loc_t, cls_t, mask_t, indices



@curry
def inference_rois(loc_p, cls_p, anchors, iou_threshold=0.5, topk=100, conf_strategy='softmax'):
    if conf_strategy == 'softmax':
        scores = torch.softmax(cls_p, dim=1)
    else:
        scores = torch.sigmoid_(cls_p)
    scores = scores[..., 1:]
    scores = torch.max(scores, dim=-1)[0]

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


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return _concat(xs, dim=0)


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
    def __init__(self, pos_thresh=0.5, mask_size=(14, 14), n_samples=None, pos_neg_ratio=1 / 3):
        super().__init__()
        self.pos_thresh = pos_thresh
        self.mask_size = mask_size
        self.n_samples = n_samples
        self.pos_neg_ratio = pos_neg_ratio

    def __call__(self, rois, image_gts):
        is_cpu = rois.device.type != 'cpu'
        match_func = match_rois if is_cpu else match_rois2
        rois_ltrb = rois[..., 1:]

        loc_targets = []
        cls_targets = []
        mask_targets = []
        sampled_rois = []
        for i in range(len(rois)):
            loc_t, cls_t, mask_t, indices = match_func(
                image_gts[i], rois_ltrb[i], self.pos_thresh, self.mask_size, self.n_samples, self.pos_neg_ratio)
            loc_targets.append(loc_t)
            cls_targets.append(cls_t)
            mask_targets.append(mask_t)
            sampled_rois.append(rois[i][indices])
        loc_t = torch.cat(loc_targets, dim=0)
        cls_t = torch.cat(cls_targets, dim=0)
        mask_t = torch.cat(mask_targets, dim=0)
        rois = torch.cat(sampled_rois, dim=0)

        return loc_t, cls_t, mask_t, rois


class MaskRCNNLoss(nn.Module):

    def __init__(self, p=0.01, rpn_cls_loss='softmax'):
        super().__init__()
        self.rpn_loss = MultiBoxLoss(p=p, cls_loss=rpn_cls_loss, prefix='RPN')
        self.rcnn_loss = MultiBoxLoss(p=p, prefix='RCNN')

    @property
    def p(self):
        return self.rcnn_loss.p

    @p.setter
    def p(self, new_p):
        self.rpn_loss.p = new_p
        self.rcnn_loss.p = new_p

    def forward(self, loc_p, cls_p, mask_p, loc_t, cls_t, mask_t, rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore):
        rpn_loc_p = rpn_loc_p.view(-1, 4)
        rpn_cls_p = rpn_cls_p.view(-1, rpn_cls_p.size(-1))
        rpn_loss = self.rpn_loss(rpn_loc_p, rpn_cls_p, rpn_loc_t, rpn_cls_t, ignore)

        num_classes = cls_p.size(-1) - 1
        loc_p = expand_last_dim(loc_p, num_classes, 4)

        pos = cls_t != 0
        labels = cls_t[pos] - 1
        loc_p = select(loc_p[pos], 1, labels)

        rcnn_loss = self.rcnn_loss(loc_p, cls_p, loc_t, cls_t)

        mask_p = select(mask_p, 1, labels)
        mask_loss = F.binary_cross_entropy_with_logits(mask_p, mask_t)
        if random.random() < self.p:
            print("mask: %.4f" % mask_loss.item())
        loss = rpn_loss + rcnn_loss + mask_loss
        return loss


@curry
def roi_based_inference(
        rois, loc_p, cls_p, predict_mask,
        iou_threshold=0.5, topk=100, nms_method='soft_nms'):
    scores, labels = torch.softmax(cls_p, dim=1)[:, 1:].max(dim=1)
    num_classes = cls_p.size(1) - 1
    loc_p = expand_last_dim(loc_p, num_classes, 4)
    loc_p = select(loc_p, 1, labels)

    loc_p[..., :2].mul_(rois[:, 2:]).add_(rois[:, :2])
    loc_p[..., 2:].exp_().mul_(rois[:, 2:])

    bboxes = loc_p

    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
    scores = scores.cpu()

    if nms_method == 'nms':
        indices = nms(bboxes, scores, iou_threshold)
        if len(indices) > topk:
            indices = indices[scores[indices].topk(topk)[1]]
        else:
            warnings.warn("Only %d RoIs left after nms rather than top %d" % (len(scores), topk))
    else:
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk)
    bboxes = BBox.convert(
        bboxes, format=BBox.LTRB, to=BBox.LTWH, inplace=True)

    if predict_mask is not None:
        mask_p = predict_mask(indices)
        masks = (select(mask_p, 1, labels[indices]).sigmoid_() > 0.5).cpu().numpy()

    dets = []
    for i, ind in enumerate(indices):
        det = {
            'image_id': -1,
            'category_id': labels[ind].item() + 1,
            'bbox': bboxes[ind].tolist(),
            'score': scores[ind].item(),
        }
        if predict_mask:
            det['segmentation'] = masks[i]
        dets.append(det)
    return dets


class RoIBasedInference:

    def __init__(self, iou_threshold=0.5, topk=100, nms_method='soft_nms'):
        self.iou_threshold = iou_threshold
        self.topk = topk
        self.nms_method = nms_method

    def __call__(self, rois, loc_p, cls_p, predict_mask):
        image_dets = []
        batch_size, num_rois = rois.size()[:2]
        rois = BBox.convert(rois, BBox.LTRB, BBox.XYWH, inplace=True)
        loc_p = loc_p.view(batch_size, num_rois, -1)
        cls_p = cls_p.view(batch_size, num_rois, -1)
        for i in range(batch_size):
            dets = roi_based_inference(
                rois[i], loc_p[i], cls_p[i], lambda indices: predict_mask(i, indices),
                self.iou_threshold, self.topk, self.nms_method)
            image_dets.append(dets)
        return image_dets
