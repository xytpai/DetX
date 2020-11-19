import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import detx_ext_cuda
import math


def assign_box(box, ph, pw, stride, size_min, size_max, radius=1.5):
    # box F(n, 4) ymin, xmin, ymax, xmax
	# ->target: L(ph, pw) 0~n-1, bg:-1
    assert box.is_cuda
    n, c = box.shape
    assert c == 4
    assert box.dtype == torch.float
    if n==0:
        return torch.full((ph, pw), -1, dtype=torch.long, device=box.device)
    box = box.view(1, 1, n, 4).expand(ph, pw, n, 4)
    output = torch.empty(ph, pw, dtype=torch.long, device=box.device)
    detx_ext_cuda.assign_box(box, stride, size_min, size_max, radius, output)
    return output


class _sigmoid_focal_loss(Function):
    @staticmethod
    def forward(ctx, input, target, gamma=2.0, alpha=0.25):
        ctx.save_for_backward(input, target)
        num_classes = input.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha
        return detx_ext_cuda.sigmoid_focal_loss_forward(input, target, 
                    num_classes, gamma, alpha)
    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        input, target = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_input = detx_ext_cuda.sigmoid_focal_loss_backward(input, target, d_loss,
                                                    num_classes, gamma, alpha)
        return d_input, None, None, None, None
sigmoid_focal_loss = _sigmoid_focal_loss.apply


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha)
        return loss.sum()


class _roi_align_corners(Function):
    '''
    Param:
    features:      F(c, ph, pw)
    rois:          F(n, 4) ymin, xmin, ymax, xmax
    feature_stride int
    out_size:      int

    Return:        F(n, c, out_size, out_size)
    '''
    @staticmethod
    def forward(ctx, features, rois, feature_stride, out_size):
        assert features.is_cuda
        assert rois.is_cuda
        assert features.dtype == torch.float
        assert rois.dtype == torch.float
        c, ph, pw = features.shape
        n, yxyx = rois.shape
        assert yxyx == 4
        ctx.rois = rois
        ctx.feature_stride = feature_stride
        ctx.out_size = out_size
        ctx.ph = ph
        ctx.pw = pw
        return detx_ext_cuda.roi_align_corners_forward(
            features, rois, feature_stride, out_size)
    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return detx_ext_cuda.roi_align_corners_backward(
            grad, ctx.rois, ctx.ph, ctx.pw, ctx.feature_stride, ctx.out_size), \
                None, None, None
roi_align_corners = _roi_align_corners.apply
