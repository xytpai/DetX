import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *


class BiFPN4(nn.Module):
    def __init__(self, in_channels=[512,1024,2048], out_channel=256):
        super().__init__()
        Conv2d = conv_with_kaiming_uniform(use_gn=False, use_relu=False)
        self.proj3 = Conv2d(in_channels[0], out_channel, kernel_size=1)
        self.proj4 = Conv2d(in_channels[1], out_channel, kernel_size=1)
        self.proj5 = Conv2d(in_channels[2], out_channel, kernel_size=1)
        self.conv3 = Conv2d(out_channel, out_channel, kernel_size=3)
        self.conv4 = Conv2d(out_channel, out_channel, kernel_size=3)
        self.conv5 = Conv2d(out_channel, out_channel, kernel_size=3)
        self.conv6 = nn.Conv2d(out_channel, out_channel, 3, 2, 1)
        nn.init.kaiming_uniform_(self.conv6.weight, a=1)
        nn.init.constant_(self.conv6.bias, 0)
        self.puri3 = Conv2d(out_channel, out_channel, kernel_size=1)
        self.puri4 = Conv2d(out_channel, out_channel, kernel_size=1)
        self.puri5 = Conv2d(out_channel, out_channel, kernel_size=1)
        
    def forward(self, x):
        c3, c4, c5 = x
        p5 = self.proj5(c5)
        p4 = self.proj4(c4)
        p3 = self.proj3(c3)

        p4 = p4 + bilinear_interpolate(p5, (p4.shape[2], p4.shape[3]))
        p3 = p3 + bilinear_interpolate(p4, (p3.shape[2], p3.shape[3]))

        p3 = self.conv3(p3)
        p4 = self.conv4(p4)
        p5 = self.conv5(p5)
        p6 = self.conv6(p5)

        n3 = p3
        n4 = p4 + bilinear_interpolate(self.puri3(n3), (p4.shape[2], p4.shape[3]))
        n5 = p5 + bilinear_interpolate(self.puri4(n4), (p5.shape[2], p5.shape[3]))
        n6 = p6 + bilinear_interpolate(self.puri5(n5), (p6.shape[2], p6.shape[3]))

        return n3, n4, n5, n6
