#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor PSRoIAlign_forward(
    const at::Tensor &input,  // Input feature map.
    const at::Tensor &rois,   // List of ROIs to pool over.
    const float scale_h,      // The scale of the image features. ROIs will be
    const float scale_w,      // scaled to this.
    const int out_channels,   // The number of output channels.
    const int pooled_height,  // The height of the pooled feature map.
    const int pooled_width,   // The width of the pooled feature
    const int sampling_ratio) // The number of points to sample in each bin
                              // along each axis.
{
    if (input.type().is_cuda()) {
#ifdef WITH_CUDA
        return PSRoIAlign_forward_cuda(input, rois, scale_h, scale_w,
                                       out_channels, pooled_height,
                                       pooled_width, sampling_ratio);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return PSRoIAlign_forward_cpu(input, rois, scale_h, scale_w, out_channels,
                                  pooled_height, pooled_width, sampling_ratio);
};

at::Tensor PSRoIAlign_backward(const at::Tensor &grad, const at::Tensor &rois,
                               const float scale_h, const float scale_w,
                               const int out_channels, const int pooled_height,
                               const int pooled_width, const int batch_size,
                               const int channels, const int height,
                               const int width, const int sampling_ratio) {
    if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
        return PSRoIAlign_backward_cuda(
            grad, rois, scale_h, scale_w, out_channels, pooled_height,
            pooled_width, batch_size, channels, height, width, sampling_ratio);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return PSRoIAlign_backward_cpu(grad, rois, scale_h, scale_w, out_channels,
                                   pooled_height, pooled_width, batch_size,
                                   channels, height, width, sampling_ratio);
};