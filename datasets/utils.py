import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage
import torch
import warnings
import torchvision.transforms as transforms
import random
warnings.filterwarnings("ignore")


def filter_annotation(anno, class_id_set, height, width, hw_th=1, area_th=1):
    anno = [obj for obj in anno if not obj.get('ignore', False)]
    anno = [obj for obj in anno if obj['iscrowd'] == 0] # filter crowd annotations
    anno = [obj for obj in anno if obj['area'] >= area_th]
    anno = [obj for obj in anno if all(o >= hw_th for o in obj['bbox'][2:])]
    anno = [obj for obj in anno if obj['category_id'] in class_id_set]
    _anno = []
    for obj in anno:
        xmin, ymin, w, h = obj['bbox']
        inter_w = max(0, min(xmin + w, width) - max(xmin, 0))
        inter_h = max(0, min(ymin + h, height) - max(ymin, 0))
        if inter_w * inter_h > 0: _anno.append(obj)
    return _anno


def x_flip(img, boxes=None, masks=None):
    # return:
    # img:   PIL
    # boxes: arr(n, 4) or None : ymin, xmin, ymax, xmax
    # masks: arr(n, h, w) or None
    img = img.transpose(Image.FLIP_LEFT_RIGHT) 
    w = img.width
    if boxes is not None and boxes.shape[0] != 0:
        xmin = w - boxes[:, 3] - 1
        xmax = w - boxes[:, 1] - 1
        boxes[:, 1] = xmin
        boxes[:, 3] = xmax
    if masks is not None and masks.shape[0] != 0:
        masks = masks[:, :, ::-1]
    return img, boxes, masks


def resize_img(img, min_size=641, max_size=1281, pad_n=64, boxes=None, masks=None):
    w, h = img.size
    smaller_size, larger_size = min(w, h), max(w, h)
    scale = min_size / float(smaller_size)
    if larger_size * scale > max_size:
        scale = max_size / float(larger_size)
    ow = round(w*scale)
    oh = round(h*scale)
    img = img.resize((ow, oh), Image.BILINEAR)
    pad_w, pad_h = (pad_n - ow % pad_n) + 1, (pad_n - oh % pad_n) + 1
    if pad_w >= pad_n: pad_w -= pad_n
    if pad_h >= pad_n: pad_h -= pad_n
    img = img.crop((0, 0, ow + pad_w, oh + pad_h))
    location = torch.FloatTensor([0, 0, oh-1, ow-1, h, w])
    if boxes is not None and boxes.shape[0] != 0:
        boxes = boxes*scale
    if masks is not None and masks.shape[0] != 0:
        masks = scipy.ndimage.zoom(masks, zoom=[1, scale, scale], order=0)
        masks_tmp = np.zeros((masks.shape[0], oh + pad_h, ow + pad_w))
        masks_tmp[:, :masks.shape[1], :masks.shape[2]] = masks
        masks = masks_tmp
    # for safe
    if boxes is not None:
        boxes[:, :2].clamp_(min=0)
        boxes[:, 2].clamp_(max=oh-2)
        boxes[:, 3].clamp_(max=ow-2)
        ymin_xmin, ymax_xmax = boxes.split([2, 2], dim=1)
        h_w = ymax_xmax - ymin_xmin + 1
        m = h_w.min(dim=1)[0] <= 1
        ymax_xmax[m] = ymin_xmin[m] + 1
        boxes = torch.cat([ymin_xmin, ymax_xmax], dim=1)
    return img, location, boxes, masks


COLOR_TABLE = [
    (256,0,0), (0,256,0), (0,0,256), 
    (255,0,255), (255,106,106),(139,58,58),(205,51,51),
    (139,0,139),(139,0,0),(144,238,144),(0,139,139)
] * 100


def draw_bbox_text(drawObj, ymin, xmin, ymax, xmax, text, color, bd=1):
    drawObj.rectangle((xmin, ymin, xmax, ymin+bd), fill=color)
    drawObj.rectangle((xmin, ymax-bd, xmax, ymax), fill=color)
    drawObj.rectangle((xmin, ymin, xmin+bd, ymax), fill=color)
    drawObj.rectangle((xmax-bd, ymin, xmax, ymax), fill=color)
    strlen = len(text)
    drawObj.rectangle((xmin, ymin, xmin+strlen*6+5, ymin+12), fill=color)
    drawObj.text((xmin+3, ymin), text)
