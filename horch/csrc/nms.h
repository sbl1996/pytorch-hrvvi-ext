#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

at::Tensor nms(const at::Tensor &dets, const at::Tensor &scores,
               const float threshold) {
    if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
        if (dets.numel() == 0) {
            //      at::cuda::CUDAGuard device_guard(dets.device());
            return at::empty({0}, dets.options().dtype(at::kLong));
        }
        auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
        return nms_cuda(b, threshold);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }

    at::Tensor result = nms_cpu(dets, scores, threshold);
    return result;
}

at::Tensor soft_nms(const at::Tensor &dets, at::Tensor &scores,
                    const float iou_threshold, const int topk,
                    const float score_threshold) {
    return soft_nms_cpu(dets, scores, iou_threshold, topk, score_threshold);
}

at::Tensor softer_nms(at::Tensor &dets, at::Tensor &scores,
                      const at::Tensor &vars, const float iou_threshold,
                      const int topk, const float sigma,
                      const float min_score) {
    return softer_nms_cpu(dets, scores, vars, iou_threshold, topk, sigma,
                          min_score);
}