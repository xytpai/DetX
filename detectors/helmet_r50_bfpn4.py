import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
from detectors.backbones import *
from detectors.necks import *
from detectors.heads import *
import detectors.cocobox_r50_bfpn4.Detector as BaseDetector
import math


class Detector(nn.Module):
    def __init__(self, cfg, mode='TEST'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.register_buffer('trained_log', torch.zeros(2).long())
        self.base_detector = BaseDetector(cfg='configs/cocobox_r50_base.yaml', mode=mode)
        if self.mode == 'TRAIN' and self.cfg['TRAIN']['BACKBONE_PRETRAINED']:
            self.backbone.load_pretrained_params(path='weights/cocobox_r50_bfpn4')
        self.base_detector.num_class    = self.cfg['DETECTOR']['NUM_CLASS']
        self.base_detector.win_minmax   = self.cfg['DETECTOR']['WIN_MINMAX']
        self.base_detector.numdets      = self.cfg['DETECTOR']['NUMDETS']
        self.base_detector.cfg          = self.cfg
        self.base_detector.bbox_head.conv_cls[-1] = \
            nn.Conv2d(channels, self.base_detector.num_class, 
            kernel_size=3, padding=1)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.base_detector.bbox_head.conv_cls[-1].bias, _bias)
        self.backbone = self.base_detector.backbone
        
    def forward(self, imgs, locations, label_cls=None, label_reg=None):
        return self.base_detector(imgs, locations, label_cls, label_reg)
