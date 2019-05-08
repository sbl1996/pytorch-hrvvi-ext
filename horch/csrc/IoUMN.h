#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor iou_mn_forward(const at::Tensor &boxes1, const at::Tensor &boxes2) {
    if (boxes1.type().is_cuda()) {
#ifdef WITH_CUDA
        return iou_mn_forward_cuda(boxes1, boxes2);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return iou_mn_forward_cpu(boxes1, boxes2);
};

std::tuple<at::Tensor, at::Tensor> iou_mn_backward(const at::Tensor &dout,
                                                   const at::Tensor &boxes1,
                                                   const at::Tensor &boxes2,
                                                   const at::Tensor &ious) {
    if (dout.type().is_cuda()) {
#ifdef WITH_CUDA
        return iou_mn_backward_cuda(dout, boxes1, boxes2, ious);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return iou_mn_backward_cpu(dout, boxes1, boxes2, ious);
};