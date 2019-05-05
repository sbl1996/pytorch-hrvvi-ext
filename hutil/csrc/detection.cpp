#include "PSROIAlign.h"
#include "ROIAlign.h"
#include "IoUMN.h"
#include <torch/extension.h>

template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor &dets, const at::Tensor &scores,
                          const float threshold) {
    AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
    AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
    AT_ASSERTM(dets.type() == scores.type(),
               "dets should have the same type as scores");

    if (dets.numel() == 0)
        return torch::empty({0}, at::device(at::kCPU).dtype(at::kLong));

    auto x1_t = dets.select(1, 0).contiguous();
    auto y1_t = dets.select(1, 1).contiguous();
    auto x2_t = dets.select(1, 2).contiguous();
    auto y2_t = dets.select(1, 3).contiguous();

    at::Tensor areas_t = (x2_t - x1_t + 0.01) * (y2_t - y1_t + 0.01);

    auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

    auto ndets = dets.size(0);
    at::Tensor suppressed_t =
        torch::zeros({ndets}, at::device(at::kCPU).dtype(at::kByte));

    auto suppressed = suppressed_t.data<uint8_t>();
    auto order = order_t.data<int64_t>();
    auto x1 = x1_t.data<scalar_t>();
    auto y1 = y1_t.data<scalar_t>();
    auto x2 = x2_t.data<scalar_t>();
    auto y2 = y2_t.data<scalar_t>();
    auto areas = areas_t.data<scalar_t>();

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);

            auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + static_cast<scalar_t>(0.01));
            auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + static_cast<scalar_t>(0.01));
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr >= threshold)
                suppressed[j] = 1;
        }
    }
    return torch::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor nms_cpu(const at::Tensor &dets, const at::Tensor &scores,
                   const float threshold) {

    auto result = torch::empty({0}, dets.type());

    AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
        result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
    });
    return result;
}

template <typename scalar_t>
at::Tensor soft_nms_cpu_kernel(const at::Tensor &dets_t, at::Tensor &scores_t,
                               const float iou_threshold, const int topk,
                               const float score_threshold) {
    AT_ASSERTM(!dets_t.type().is_cuda(), "dets_t must be a CPU tensor");
    AT_ASSERTM(!scores_t.type().is_cuda(), "scores must be a CPU tensor");
    AT_ASSERTM(dets_t.type() == scores_t.type(),
               "dets_t should have the same type as scores");

    if (dets_t.numel() == 0)
        return torch::empty({0}, at::device(at::kCPU).dtype(at::kLong));

    auto x1_t = dets_t.select(1, 0).contiguous();
    auto y1_t = dets_t.select(1, 1).contiguous();
    auto x2_t = dets_t.select(1, 2).contiguous();
    auto y2_t = dets_t.select(1, 3).contiguous();

    at::Tensor areas_t = (x2_t - x1_t + 0.01) * (y2_t - y1_t + 0.01);

    auto ndets = dets_t.size(0);
    at::Tensor suppressed_t =
        torch::zeros({ndets}, at::device(at::kCPU).dtype(at::kByte));

    auto suppressed = suppressed_t.data<uint8_t>();
    auto x1 = x1_t.data<scalar_t>();
    auto y1 = y1_t.data<scalar_t>();
    auto x2 = x2_t.data<scalar_t>();
    auto y2 = y2_t.data<scalar_t>();
    auto areas = areas_t.data<scalar_t>();
    auto scores = scores_t.data<scalar_t>();
    std::vector<scalar_t> indices;

    for (int64_t _i = 0; _i < topk; _i++) {
        scalar_t max_score = 0;
        int64_t i = -1;
        for (int64_t j = 0; j < ndets; j++) {
            if (suppressed[j] == 1)
                continue;
            if (scores[j] >= max_score) {
                max_score = scores[j];
                i = j;
            }
        }
        if (i == -1)
            break;
        indices.push_back(i);
        suppressed[i] = 1;

        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t j = 0; j < ndets; j++) {
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);

            auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + static_cast<scalar_t>(0.01));
            auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + static_cast<scalar_t>(0.01));
            auto inter = w * h;
            auto iou = inter / (iarea + areas[j] - inter);
            if (iou >= iou_threshold) {
                scores[j] *= 1 - iou;
                if (scores[j] < score_threshold) {
                    suppressed[j] = 1;
                }
            }
        }
    }
    auto n = static_cast<int64_t>(indices.size());
    at::Tensor indices_t =
        torch::empty({n}, at::device(at::kCPU).dtype(at::kLong));
    auto indices_p = indices_t.data<int64_t>();
    for (auto i = 0; i < n; i++)
        indices_p[i] = indices[i];
    return indices_t;
}

at::Tensor soft_nms_cpu(const at::Tensor &dets, at::Tensor &scores,
                        const float iou_threshold, const int topk,
                        const float score_threshold) {

    auto result = torch::empty({0}, dets.type());

    AT_DISPATCH_FLOATING_TYPES(dets.type(), "soft_nms_cpu", [&] {
        result = soft_nms_cpu_kernel<scalar_t>(dets, scores, iou_threshold,
                                               topk, score_threshold);
    });
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_cpu", &nms_cpu, "nms_cpu");
    m.def("soft_nms_cpu", &soft_nms_cpu, "soft_nms_cpu");
    m.def("iou_mn_forward", &iou_mn_forward, "iou_mn_forward");
    m.def("iou_mn_backward", &iou_mn_backward, "iou_mn_backward");
    m.def("psroi_align_forward", &PSROIAlign_forward, "PSROIAlign_forward");
    m.def("psroi_align_backward", &PSROIAlign_backward, "PSROIAlign_backward");
    m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
    m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
}