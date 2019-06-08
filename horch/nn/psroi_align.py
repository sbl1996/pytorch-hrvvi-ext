import torch
from torch import nn

from torch.autograd import Function
from torch.autograd.function import once_differentiable

from torch.nn.modules.utils import _pair

from horch import _C


class _PSROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, out_channels, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = output_size
        ctx.out_channels = out_channels
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.psroi_align_forward(
            input, roi, spatial_scale[0], spatial_scale[1], out_channels,
            output_size[0], output_size[1], sampling_ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        out_channels = ctx.out_channels
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.psroi_align_backward(
            grad_output, rois, spatial_scale[0], spatial_scale[1], out_channels,
            output_size[0], output_size[1], bs, ch, h, w, sampling_ratio)
        return grad_input, None, None, None, None, None


psroi_align = _PSROIAlign.apply


class PSRoIAlign(nn.Module):
    def __init__(self, out_channels, output_size, spatial_scale=None, sampling_ratio=2, adaptive=True):
        super().__init__()
        self.out_channels = out_channels
        self.output_size = _pair(output_size)
        self.spatial_scale = _pair(spatial_scale)
        self.sampling_ratio = sampling_ratio
        self.adaptive = adaptive

    def forward(self, input, rois):
        spatial_scale = self.spatial_scale
        if self.adaptive:
            spatial_scale = tuple(input.size()[2:4])
        assert rois.size(-1) == 5, "Batch indices must be provided."
        rois = rois.view(-1, 5)
        return psroi_align(input, rois, self.out_channels, self.output_size, spatial_scale, self.sampling_ratio)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'out_channels=' + str(self.out_channels)
        tmpstr += ', output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', adaptive=' + str(self.adaptive)
        tmpstr += ')'
        return tmpstr