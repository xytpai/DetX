import torch
import torch.nn as nn 
import torch.nn.functional as F
from .misc import box_iou


def cluster_nms(classes, scores, boxes, nms_iou, other=[]):
    scores, idx = scores.sort(0, descending=True)
    boxes = boxes[idx]
    classes = classes[idx]
    mask = (classes.view(-1,1) == classes.view(1,-1)).float()
    iou = box_iou(boxes, boxes)*mask
    iou.triu_(diagonal=1)
    C = iou
    for i in range(200):
        A = C
        maxA = A.max(dim=0)[0]
        E = (maxA < nms_iou).float().unsqueeze(1).expand_as(A)
        C = iou.mul(E)
        if A.equal(C): break
    keep = maxA < nms_iou
    if len(other) > 0:
        other = [other[i][idx][keep] for i in range(len(other))]
    return classes[keep], scores[keep], boxes[keep], other
