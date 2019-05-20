from toolz import curry

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.detection.one import MultiBoxLoss, coords_to_target, target_to_coords, match_anchors_flat, AnchorBasedInference
from horch.detection.bbox import BBox
from horch.detection.iou import iou_mn
from horch.detection.nms import nms, soft_nms_cpu


def match_rois(anns, rois, rois_xywh, pos_thresh=0.5):
    num_rois = len(rois)
    loc_t = rois.new_zeros(num_rois, 4)
    cls_t = loc_t.new_zeros(num_rois, dtype=torch.long)

    if len(anns) == 0:
        return loc_t, cls_t

    bboxes = loc_t.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = loc_t.new_tensor([ann['category_id'] for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, rois)

    max_ious, indices = ious.max(dim=1)
    loc_t[indices] = coords_to_target(bboxes, rois_xywh[indices])
    cls_t[indices] = labels

    pos = ious > pos_thresh
    for ipos, bbox, label in zip(pos, bboxes, labels):
        loc_t[ipos] = coords_to_target(bbox, rois_xywh[ipos])
        cls_t[ipos] = label

    target = [loc_t, cls_t]
    return target


class MatchRoIs:
    def __init__(self, pos_thresh=0.5):
        super().__init__()
        self.pos_thresh = pos_thresh

    def __call__(self, rois, image_gts):
        batch_size, num_rois = rois.size()[:2]
        rois_xywh = BBox.convert(rois, format=BBox.LTRB, to=BBox.XYWH)
        loc_targets = []
        cls_targets = []
        for i in range(batch_size):
            loc_t, cls_t = match_rois(image_gts[i], rois[i], rois_xywh[i], self.pos_thresh)
            loc_targets.append(loc_t)
            cls_targets.append(cls_t)
        loc_t = torch.cat(loc_targets, dim=0)
        cls_t = torch.cat(cls_targets, dim=0)
        return loc_t, cls_t


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return torch.cat(xs, dim=0)


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
        target.append(anns)
        return img, target


class RCNNLoss(nn.Module):

    def __init__(self, p=0.01):
        super().__init__()
        self.roi_match = MatchRoIs(pos_thresh=0.5)
        self.criterion = MultiBoxLoss(p=p)

    @property
    def p(self):
        return self.criterion.p

    @p.setter
    def p(self, new_p):
        self.criterion.p = new_p

    def forward(self, loc_p, cls_p, rois, image_gts):
        loc_t, cls_t = self.roi_match(rois, image_gts)
        loss = self.criterion(loc_p, cls_p, loc_t, cls_t)
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
        mask = scores > conf_threshold
        scores = scores[mask]
        labels = labels[mask]
        bboxes = bboxes[mask]
        rois = rois[mask]

    bboxes = target_to_coords(bboxes, rois)

    bboxes = BBox.convert(
        bboxes, format=BBox.XYWH, to=BBox.LTRB, inplace=True).cpu()
    scores = scores.cpu()

    if nms == 'nms':
        indices = nms(bboxes, scores, iou_threshold)
        scores = scores[indices]
        labels = labels[indices]
        bboxes = bboxes[indices]
        if scores.size(0) > topk:
            indices = scores.topk(topk)[1]
        else:
            indices = range(scores.size(0))
    else:
        indices = soft_nms_cpu(
            bboxes, scores, iou_threshold, topk, min_score=conf_threshold)
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

    def __call__(self, loc_p, cls_p, rois):
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


