import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
from detectors.backbones import *
from detectors.necks import *
from detectors.heads import *


class Detector(nn.Module):
    def __init__(self, cfg, mode='TEST'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.register_buffer('trained_log', torch.zeros(2).long())
        self.num_class    = self.cfg['DETECTOR']['NUM_CLASS']
        self.win_minmax   = self.cfg['DETECTOR']['WIN_MINMAX']
        self.numdets      = self.cfg['DETECTOR']['NUMDETS']
        self.backbone     = ResNet(self.cfg['DETECTOR']['DEPTH'])
        self.neck         = BiFPN4(self.backbone.out_channels, 256)
        self.bbox_head    = BBOXHead(256, self.num_class)
        self.focal_loss   = SigmoidFocalLoss(2.0, 0.25)
        self.iou_loss     = IOULoss()
        if self.mode == 'TRAIN' and self.cfg['TRAIN']['BACKBONE_PRETRAINED']:
            self.backbone.load_pretrained_params()
        
    def forward(self, imgs, locations, label_cls=None, label_reg=None):
        '''
        imgs:       F(b, 3, im_h, im_w)
        locations:  F(b, 6)       yxyx ori_h ori_w
        label_cls:  L(b, n)       0:pad
        label_reg:  F(b, n, 4)    yxyx
        '''
        batch_size, _, im_h, im_w = imgs.shape
        out_ft = self.neck(self.backbone(imgs))
        pred_cls, pred_reg, pred_info = [], [], []
        for ft in out_ft:
            _, _, ft_h, ft_w = ft.shape
            cls_s, reg_s = self.bbox_head(ft, im_h, im_w)
            pred_cls.append(cls_s)
            pred_reg.append(reg_s)
            pred_info.append([ft_h, ft_w, (im_h-1)//(ft_h-1)])
        if label_cls is not None and label_reg is not None:
            return self._loss(locations, im_h, im_w, pred_cls, pred_reg, pred_info, \
                                label_cls, label_reg)
        else:
            return self._pred(locations, im_h, im_w, pred_cls, pred_reg, pred_info)
    
    def _loss(self, locations, im_h, im_w, pred_cls, pred_reg, pred_info, 
                label_cls, label_reg):
        pred_cls, pred_reg = torch_cat([pred_cls, pred_reg], dim=1)
        loss = []
        for b in range(pred_cls.shape[0]):
            # filter out padding labels
            label_cls_b, label_reg_b = label_cls[b], label_reg[b]
            m = label_cls_b > 0
            label_cls_b, label_reg_b = label_cls_b[m], label_reg_b[m]
            # get target
            target_idx_b = []
            for s in range(len(pred_info)):
                ft_h, ft_w, ft_stride = pred_info[s]
                target_idx_b.append(assign_box(label_reg_b, ft_h, ft_w, ft_stride,
                    self.win_minmax[s][0], self.win_minmax[s][1]).view(-1))
            target_idx_b = torch.cat(target_idx_b, dim=0)
            m_pos = target_idx_b >= 0  # B(an)
            num_pos = float(m_pos.sum())
            target_cls_selected = label_cls_b[target_idx_b] # L(an)
            target_cls_selected[~m_pos] = 0
            pred_cls_selected = pred_cls[b] # F(an, num_class)
            loss_cls = self.focal_loss(pred_cls_selected, target_cls_selected).view(1)
            if num_pos <= 0: # no object assigned
                loss.append(loss_cls)
                continue
            target_reg_selected = label_reg_b[target_idx_b[m_pos]]
            pred_reg_selected = pred_reg[b][m_pos]
            loss_reg = self.iou_loss(pred_reg_selected, target_reg_selected).view(1)            
            loss.append((loss_cls+loss_reg)/num_pos)
        return torch.cat(loss)

    def _pred(self, locations, im_h, im_w, pred_cls, pred_reg, pred_info):
        assert self.mode != 'TRAIN'
        batch_size = pred_cls[0].shape[0]
        assert batch_size == 1
        _pred_cls_i, _pred_cls_p, _pred_reg = [], [], []
        for s in range(len(pred_info)):
            ft_h, ft_w, ft_stride = pred_info[s]
            pred_cls_p_s, pred_cls_i_s = torch.max(pred_cls[s][0].sigmoid(), dim=1)
            pred_cls_i_s = pred_cls_i_s + 1
            m = pred_cls_p_s > self.cfg[self.mode]['NMS_TH']
            pred_cls_i_s, pred_cls_p_s, pred_reg_s = torch_select(
                [pred_cls_i_s, pred_cls_p_s, pred_reg[s][0]], m)
            nms_maxnum = min(int(self.cfg[self.mode]['NMS_TOPK_P']), pred_cls_p_s.shape[0])
            select = torch.topk(pred_cls_p_s, nms_maxnum, largest=True, dim=0)[1]
            _pred_cls_i.append(pred_cls_i_s[select])
            _pred_cls_p.append(pred_cls_p_s[select])
            _pred_reg.append(pred_reg_s[select])
        pred_cls_i, pred_cls_p, pred_reg = torch_cat(
            [_pred_cls_i, _pred_cls_p, _pred_reg], dim=0)
        # throw none
        if pred_cls_i.shape[0] == 0:
            return {
                'bbox': torch.empty(0, 4).float(),
                'class': torch.empty(0).long(),
                'score': torch.empty(0).float()
            }
        # clamp
        if locations.shape[0]==1: locations = locations[0]
        valid_ymin, valid_xmin, valid_ymax, valid_xmax, ori_h, ori_w = \
            float(locations[0]), float(locations[1]), float(locations[2]), \
            float(locations[3]), float(locations[4]), float(locations[5])
        pred_reg[..., 0].clamp_(min=valid_ymin)
        pred_reg[..., 1].clamp_(min=valid_xmin)
        pred_reg[..., 2].clamp_(max=valid_ymax)
        pred_reg[..., 3].clamp_(max=valid_xmax)
        # nms for each class
        pred_cls_i, pred_cls_p, pred_reg, _ = cluster_nms(
            pred_cls_i, pred_cls_p, pred_reg, self.cfg[self.mode]['NMS_IOU'])
        # numdets
        numdets = min(self.numdets, pred_cls_i.shape[0])
        select = torch.topk(pred_cls_p, numdets, largest=True, dim=0)[1]
        pred_cls_i, pred_cls_p, pred_reg = torch_select(
            [pred_cls_i, pred_cls_p, pred_reg], select)
        # transfer
        pred_reg = transfer_box_(pred_reg, valid_ymin, valid_xmin, 
            valid_ymax, valid_xmax, ori_h, ori_w)
        return {
            'bbox': pred_reg.cpu(),
            'class': pred_cls_i.cpu(),
            'score': pred_cls_p.cpu()
        }
    
    def load_pretrained_params(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
