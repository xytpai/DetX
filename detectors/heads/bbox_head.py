import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
import math


class BBOXHead(nn.Module):
    def __init__(self, channels, num_class, norm=4):
        super().__init__()
        self.channels = channels
        self.num_class = num_class
        self.norm = norm
        self.conv_cls = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_class, kernel_size=3, padding=1))
        self.conv_reg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 4, kernel_size=3, padding=1))
        for block in [self.conv_cls, self.conv_reg]:
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.bias, 0)
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)
    
    def forward(self, x, im_h, im_w):
        '''
        x: F(b, c, h_s, w_s)

        Return: 
        cls_s: F(b, h_s*w_s, num_class)
        reg_s: F(b, h_s*w_s, 4) ymin, xmin, ymax, xmax
        '''
        batch_size, c, h_s, w_s = x.shape
        stride = (im_h-1) // (h_s-1)
        cls_s = self.conv_cls(x)
        reg_s = self.conv_reg(x)
        cls_s = cls_s.permute(0,2,3,1).contiguous()
        reg_s = reg_s.permute(0,2,3,1).contiguous()*(self.norm*stride)
        cls_s = cls_s.view(batch_size, -1, self.num_class)
        reg_s = reg_s.view(batch_size, -1, 4)
        # transform reg
        y = torch.linspace(0, im_h-1, h_s).to(cls_s.device)
        x = torch.linspace(0, im_w-1, w_s).to(cls_s.device)
        center_y, center_x = torch.meshgrid(y, x) # F(h_s, w_s)
        center_y = center_y.contiguous()
        center_x = center_x.contiguous()
        center_yx = torch.cat([center_y.view(-1, 1), center_x.view(-1, 1)], dim=1)
        reg_s_ymin_xmin =  center_yx + reg_s[..., 0:2]
        reg_s_ymax_xmax =  center_yx + reg_s[..., 2:4]
        reg_s = torch.cat([reg_s_ymin_xmin, reg_s_ymax_xmax], dim=2)
        return cls_s, reg_s
