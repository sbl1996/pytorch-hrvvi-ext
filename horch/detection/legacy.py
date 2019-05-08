#
# def soft_nms(boxes, confidences, iou_threshold, topk=10):
#     r"""
#     Args:
#         boxes(tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
#         confidences: (N,)
#     Returns:
#         indices: (N,)
#     """
#     confidences = confidences.clone()
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#
#     zero = boxes.new_tensor(0)
#     indices = []
#
#     while True:
#         i = confidences.argmax()
#         # print(i)
#         # print(confidences[i])
#         indices.append(i)
#         if len(indices) >= topk:
#             break
#         xx1 = torch.max(x1[i], x1)
#         yy1 = torch.max(y1[i], y1)
#         xx2 = torch.min(x2[i], x2)
#         yy2 = torch.min(y2[i], y2)
#
#         w = torch.max(zero, xx2 - xx1 + 1)
#         h = torch.max(zero, yy2 - yy1 + 1)
#
#         inter = w * h
#         ious = inter / (areas[i] + areas - inter)
#         mask = ious >= iou_threshold
#         confidences[mask] *= (1 - ious)[mask]
#     return boxes.new_tensor(indices, dtype=torch.long)
#
#
# def non_max_suppression(boxes, confidences, iou_threshold=0.5):
#     r"""
#     Args:
#         boxes(tensor of shape `(N, 4)`): [xmin, ymin, xmax, ymax]
#         confidences: (N,)
#         max_boxes(int):
#         iou_threshold(float):
#     Returns:
#         indices: (N,)
#     """
#     N = len(boxes)
#     confs, orders = confidences.sort(descending=True)
#     boxes = boxes[orders]
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     suppressed = confidences.new_zeros(N, dtype=torch.uint8)
#
#     zero = boxes.new_tensor(0)
#
#     for i in range(N):
#         if suppressed[i] == 1:
#             continue
#         xx1 = torch.max(x1[i], x1[i + 1:])
#         yy1 = torch.max(y1[i], y1[i + 1:])
#         xx2 = torch.min(x2[i], x2[i + 1:])
#         yy2 = torch.min(y2[i], y2[i + 1:])
#
#         w = torch.max(zero, xx2 - xx1 + 1)
#         h = torch.max(zero, yy2 - yy1 + 1)
#
#         inter = w * h
#         iou = inter / (areas[i] + areas[i + 1:] - inter)
#         suppressed[i + 1:][iou > iou_threshold] = 1
#     return orders[torch.nonzero(suppressed == 0).squeeze(1)]
#
