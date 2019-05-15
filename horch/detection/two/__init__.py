import numpy as np
import torch

from horch._numpy import iou_mn, iou_11, iou_mm
from horch.detection import BBox


def sample(t, n):
    if len(t) >= n:
        indices = torch.randperm(len(t), device=t.device)[:n]
    else:
        indices = torch.randint(len(t), size=(n,), device=t.device)
    return t[indices]


def coords_to_target(gt_box, anchors):
    box_txty = (gt_box[..., :2] - anchors[..., :2]) / anchors[..., 2:]
    box_twth = (gt_box[..., 2:] / anchors[..., 2:]).log_()
    return torch.cat((box_txty, box_twth), dim=-1)


def batch_anchor_match(image_gts, anchors_xywh, anchors_ltrb, max_iou=True, pos_thresh=0.5, neg_thresh=None,
                       n_samples=0, pos_neg_ratio=None, debug=False, get_label=lambda x: x['category_id']):
    loc_t = []
    cls_t = []
    neg_or_indices = []
    batch_size = len(image_gts)
    for i in range(batch_size):
        target = match_anchors(
            image_gts[i], anchors_xywh[i], anchors_ltrb[i],
            max_iou, pos_thresh, neg_thresh, n_samples, pos_neg_ratio, debug, get_label)
        if len(target) == 2:
            i_loc_t, i_cls_t = target
            loc_t.append(i_loc_t)
            cls_t.append(i_cls_t)
        else:
            i_loc_t, i_cls_t, neg_or_ind = target
            loc_t.append(i_loc_t)
            cls_t.append(i_cls_t)
            neg_or_indices.append(neg_or_ind)

    loc_t = torch.stack(loc_t, dim=0)
    cls_t = torch.stack(cls_t, dim=0)
    if len(neg_or_indices) != 0:
        neg_or_indices = torch.stack(neg_or_indices, dim=0)
        return loc_t, cls_t, neg_or_indices
    else:
        return loc_t, cls_t


def match_anchors(anns, anchors, anchors_ltrb, max_iou=True, pos_thresh=0.5, neg_thresh=None,
                  n_samples=0, pos_neg_ratio=None, debug=False, get_label=lambda x: x['category_id']):
    num_anchors = len(anchors)
    loc_t = anchors.new_zeros(num_anchors, 4)
    cls_t = loc_t.new_zeros(num_anchors, dtype=torch.long)

    if len(anns) == 0:
        target = [loc_t, cls_t]
        if pos_thresh and neg_thresh:
            target.append(loc_t.new_zeros(num_anchors, dtype=torch.uint8))
        return target

    bboxes = loc_t.new_tensor([ann['bbox'] for ann in anns])
    bboxes = BBox.convert(bboxes, format=BBox.LTWH, to=BBox.XYWH, inplace=True)
    labels = loc_t.new_tensor([get_label(ann) for ann in anns], dtype=torch.long)

    bboxes_ltrb = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB)
    ious = iou_mn(bboxes_ltrb, anchors_ltrb)

    if max_iou:
        max_ious, indices = ious.max(dim=1)
        loc_t[indices] = coords_to_target(bboxes, anchors[indices])
        cls_t[indices] = labels
        if debug:
            print(max_ious)

    pos = ious > pos_thresh
    for ipos, bbox, label in zip(pos, bboxes, labels):
        loc_t[ipos] = coords_to_target(bbox, anchors[ipos])
        cls_t[ipos] = label

    pos = cls_t != 0
    neg = ~pos
    if neg_thresh:
        neg &= (ious > neg_thresh).sum(dim=0) == 0

    if n_samples:
        pos_frac = pos_neg_ratio / (pos_neg_ratio + 1)
        num_pos = min(n_samples * pos_frac, pos.sum().item())
        num_neg = n_samples - num_pos
        pos_indices = sample(torch.nonzero(pos), num_pos)
        neg_indices = sample(torch.nonzero(neg), num_neg)
        indices = torch.cat([pos_indices, neg_indices], dim=0)
        loc_t = loc_t[indices]
        cls_t = cls_t[indices]
        return loc_t, cls_t, indices

    if neg_thresh:
        return loc_t, cls_t, neg
    else:
        return loc_t, cls_t


def flatten(xs):
    if torch.is_tensor(xs):
        return xs.view(-1, xs.size(-1))
    xs = [x.view(-1, x.size(-1)) for x in xs]
    return torch.cat(xs, dim=0)


class AnchorMatcher:
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

    def __init__(self, max_iou=True,
                 pos_thresh=0.5, neg_thresh=None,
                 n_samples=0, pos_neg_ratio=None,
                 anchors=None, debug=False, get_label=lambda x: x['category_id']):
        if anchors is not None:
            self.anchors_xywh = flatten(anchors)
            self.anchors_ltrb = BBox.convert(self.anchors_xywh, BBox.XYWH, BBox.LTRB)
        else:
            self.anchors_xywh = self.anchors_ltrb = None
        self.max_iou = max_iou
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.n_samples = n_samples
        self.pos_neg_ratio = pos_neg_ratio
        self.get_label = get_label
        self.debug = debug

    def __call__(self, anns, anchors_xywh=None, anchors_ltrb=None):
        if len(anns) == 0:
            return
        if anchors_xywh is None:
            anchors_xywh = self.anchors_xywh
        else:
            anchors_xywh = flatten(anchors_xywh)
            if anchors_ltrb is None:
                anchors_ltrb = BBox.convert(anchors_xywh, BBox.XYWH, BBox.LTRB)
        target = match_anchors(
            anns, anchors_xywh, anchors_ltrb,
            self.max_iou, self.pos_thresh, self.neg_thresh,
            self.n_samples, self.pos_neg_ratio,
            self.debug, self.get_label)
        return target


def kmeans(X, k, max_iter=300, tol=1e-6, verbose=True):
    n, d = X.shape
    centers = X[np.random.choice(n, size=k, replace=False)]
    for i in range(max_iter):
        dist = 1 - iou_mn(X, centers)
        y = np.argmin(dist, axis=1)
        loss = 0
        for ki in range(k):
            kx = X[y == ki]
            if kx.size != 0:
                center = np.mean(kx, axis=0)
                iou = iou_11(center, centers[ki])
                loss += 1 - iou
                centers[ki] = center
        loss /= k
        if verbose:
            print("Iter %d: %.6f" % (i, loss))
        if loss < tol:
            break
    return y, centers


def find_centers_kmeans(bboxes, k, max_iter=100, verbose=True):
    r"""
    Find bounding box centers by kmeans.

    Parameters
    ----------
    bboxes : ``numpy.ndarray``
        Bounding boxes of normalized [xmin, ymin, xmax, ymax].
    k : ``int``
        Number of clusters (priors).
    max_iter : ``int``
        Maximum numer of iterations. Default: 100
    verbose: ``bool``
        Whether to print info.
    """
    centers = kmeans(bboxes, k, max_iter, verbose=verbose)[1]
    if verbose:
        mean_iou = iou_mn(bboxes, centers).max(axis=1).mean()
        print("Mean IoU: %.4f" % mean_iou)
    return centers


def find_priors_kmeans(sizes, k, max_iter=100, verbose=True):
    r"""
    Find bounding box centers by kmeans.

    Parameters
    ----------
    sizes : ``numpy.ndarray``
        Bounding boxes of normalized [width, height].
    k : ``int``
        Number of clusters (priors).
    max_iter : ``int``
        Maximum numer of iterations. Default: 100
    verbose: ``bool``
        Whether to print info.
    """
    bboxes = np.concatenate([np.full_like(sizes, 0.5), sizes], axis=-1)
    bboxes = BBox.convert(bboxes, BBox.XYWH, BBox.LTRB, inplace=True)
    centers = find_centers_kmeans(bboxes, k, max_iter, verbose)
    centers[:, 2] -= centers[:, 0]
    centers[:, 3] -= centers[:, 1]
    priors = centers[:, 2:]
    return torch.from_numpy(priors).float()
