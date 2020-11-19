import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageDraw
if __name__ != '__main__':
    from datasets.utils import *
else:
    from utils import *


class AspectRatioBasedSampler(data.Sampler):
    def __init__(self, dataset, batch_size, min_sizes, drop_last=True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.min_sizes  = min_sizes
        if min_sizes[0] < 0:
            print('using multi-scale training')
            self.random_en = True
        else: self.random_en = False
        order = list(range(len(self.dataset)))
        order.sort(key=lambda x: self.dataset.image_aspect_ratio(x))
        self.groups = [[order[x % len(order)] for x in range(i, i+self.batch_size)]
                        for i in range(0, len(order), self.batch_size)]
        if drop_last: 
            self.len = len(self.dataset) // self.batch_size
        else:
            self.len = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        self.groups = self.groups[:self.len]

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            if self.random_en: 
                min_size = random.randint(self.min_sizes[1], self.min_sizes[2])
            else:
                min_size = random.choice(self.min_sizes)
            for item in group:
                yield item, min_size


class Dataset(torchvision.datasets.coco.CocoDetection):
    name_table = ['background', 
               'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, cfg, mode='TEST'):
        # base
        self.loss_def = "self.detector(data['imgs'], data['locations'], data['labels'], data['boxes']).mean()"
        self.cfg = cfg
        self.mode = mode
        self.root_img = cfg['DATASET']['ROOT_'+mode]
        self.file_json = cfg['DATASET']['JSON_'+mode]
        super(Dataset, self).__init__(self.root_img, self.file_json)
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        # name_table
        self.index_to_coco = [i for i in range(len(self.name_table))]
        self.coco_to_index = {}
        for cate in self.coco.loadCats(self.coco.getCatIds()):
            name = cate['name']
            if name in self.name_table:
                index = self.name_table.index(name)
                self.index_to_coco[index] = cate['id']
                self.coco_to_index[cate['id']] = index
        # filter self.ids
        ids = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            height, width = img_info['height'], img_info['width']
            if min(height, width) < 32: continue
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anno = self.coco.loadAnns(ann_ids)
            if len(filter_annotation(anno, self.coco_to_index, height, width))>0:
                ids.append(img_id)
        self.ids = ids
        # TODO: additional
        self.max_size = cfg[mode]['MAX_SIZE']
        self.pad_n = cfg[mode]['PAD_N']
        self.min_size = cfg[mode].get('MIN_SIZE', 1)
        self.min_sizes = cfg[mode].get('MIN_SIZES', [self.min_size])
        self.normalize = cfg[mode].get('NORMALIZE', True)

    def __getitem__(self, data):
        '''
        Return:
        img:      F(3, h, w)
        location: F(6)
        boxes:    F(n, 4)
        labels:   L(n)
        '''
        # base
        assert self.mode == 'TRAIN'
        idx, min_size = data
        img, anno = super(Dataset, self).__getitem__(idx)
        anno = filter_annotation(anno, self.coco_to_index, img.size[1], img.size[0])
        boxes = [obj['bbox'] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        xmin_ymin, w_h = boxes.split([2, 2], dim=1)
        xmax_ymax = xmin_ymin + w_h - 1
        xmin, ymin = xmin_ymin.split([1, 1], dim=1)
        xmax, ymax = xmax_ymax.split([1, 1], dim=1)
        boxes = torch.cat([ymin, xmin, ymax, xmax], dim=1)
        labels = [self.coco_to_index[obj['category_id']] for obj in anno]
        labels = torch.LongTensor(labels)
        # clamp
        boxes[:, :2].clamp_(min=0)
        boxes[:, 2].clamp_(max=float(img.size[1])-1)
        boxes[:, 3].clamp_(max=float(img.size[0])-1)
        # transform
        if random.random() < 0.5: img, boxes, _ = x_flip(img, boxes)
        img, location, boxes, _ = resize_img(img, min_size, self.max_size, self.pad_n, boxes)
        img = transforms.ToTensor()(img)
        if self.normalize: img = self.normalizer(img)
        return img, location, boxes, labels

    def collate_fn(self, data):
        '''
        Return:
        imgs:      F(b, 3, h, w)
        locations: F(b, 6)
        boxes:     F(b, max_n, 4)
        labels:    L(b, max_n)            bg:0
        '''
        imgs, locations, boxes, labels = zip(*data)
        locations = torch.stack(locations)
        batch_num = len(imgs)
        max_h, max_w, max_n = 0, 0, 0
        for b in range(batch_num):
            if imgs[b].shape[1] > max_h: max_h = imgs[b].shape[1]
            if imgs[b].shape[2] > max_w: max_w = imgs[b].shape[2]
            if boxes[b].shape[0] > max_n: max_n = boxes[b].shape[0]
        imgs_t = torch.zeros(batch_num, 3, max_h, max_w)
        boxes_t = torch.zeros(batch_num, max_n, 4)
        labels_t = torch.zeros(batch_num, max_n).long()
        for b in range(batch_num):
            imgs_t[b, :, :imgs[b].shape[1], :imgs[b].shape[2]] = imgs[b]
            boxes_t[b, :boxes[b].shape[0]] = boxes[b]
            labels_t[b, :boxes[b].shape[0]] = labels[b]
        return {'imgs':imgs_t, 'locations':locations, 
                    'boxes':boxes_t, 'labels':labels_t}
    
    def image_aspect_ratio(self, idx):
        image = self.coco.loadImgs(self.ids[idx])[0]
        return float(image['width']) / float(image['height'])
    
    def transform_inference_img(self, img_pil):
        assert self.mode != 'TRAIN'
        if img_pil.mode != 'RGB': img_pil = img_pil.convert('RGB')
        img_pil, location, _, _ = resize_img(img_pil, self.min_size, self.max_size)
        img = transforms.ToTensor()(img_pil)
        if self.normalize: img = self.normalizer(img)
        img = img.unsqueeze(0)
        return img, location
    
    def make_loader(self):
        assert self.mode == 'TRAIN'
        batch_size = self.cfg['TRAIN']['BATCH_SIZE']
        sampler = AspectRatioBasedSampler(self, batch_size, self.min_sizes, 
                    drop_last=self.cfg['TRAIN'].get('DROP_LAST', True))
        return data.DataLoader(self, batch_size=batch_size, 
                    sampler=sampler, num_workers=self.cfg['TRAIN']['NUM_WORKERS'], 
                    collate_fn=self.collate_fn)
    
    def show(self, img, pred, file_name=None):
        '''
        img:   F(3, h, w) or PIL
        pred:  dict {
            'bbox':   F(n, 4)
            'class':  L(n) 0:bg
            'score':  F(n) or None
        }
        file_name: 'out.jpg' or None
        '''
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)
        boxes, labels, scores = pred.get('bbox', None), \
            pred.get('class', None), pred.get('score', None)
        # sort
        hw = boxes[:, 2:] - boxes[:, :2]
        area = hw[:, 0] * hw[:, 1] # N
        select = area.sort(descending=True)[1] # L(n)
        # draw bbox
        drawObj = ImageDraw.Draw(img)
        for i in range(select.shape[0]):
            i = int(select[i])
            lb = int(labels[i])
            if lb > 0: # fg
                box = boxes[i]
                if scores is None:
                    draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], self.name_table[lb],
                        color=COLOR_TABLE[i])
                else:
                    str_score = str(float(scores[i]))[:5]
                    str_out = self.name_table[lb] + ': ' + str_score
                    draw_bbox_text(drawObj, box[0], box[1], box[2], box[3], str_out, 
                        color=COLOR_TABLE[i])
        if file_name is not None: img.save(file_name)
        else: img.show()


if __name__ == '__main__':
    cfg = {
        'DATASET': {
            'ROOT_TRAIN': '..\\demo_images',
            'JSON_TRAIN': '..\\demo_images\\instances_demo.json'
        },
        'TRAIN': {
            'MIN_SIZES': [641],
            'MAX_SIZE': 1281,
            'PAD_N': 128,
            'NORMALIZE': False,
            'BATCH_SIZE': 1,
            'NUM_WORKERS': 0
        }
    }
    mode = 'TRAIN'
    dataset = Dataset(cfg, mode)
    loader = dataset.make_loader()
    for data in loader:
        imgs, locations, boxes, labels = data['imgs'], \
            data['locations'], data['boxes'], data['labels']
        print('imgs:', imgs.shape)
        print('locations:', locations.shape)
        print('boxes:', boxes.shape)
        print('labels:', labels.shape)
        b = random.randint(0, cfg['TRAIN']['BATCH_SIZE']-1)
        pred = {
            'bbox': boxes[b],
            'class': labels[b]
        }
        dataset.show(imgs[b], pred)
        break
