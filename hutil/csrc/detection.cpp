#include <torch/extension.h>
#include "PSROIAlign.h"

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

    at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

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

            auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
            auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
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

    at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

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

            auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
            auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
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

template <typename T>
void iou_mn_forward_kernel(const T *boxes1, const T *boxes2, int64_t m,
                           int64_t n, T *ious) {

    for (auto i = 0; i < m; i++) {
        auto ibox = boxes1 + 4 * i;
        auto ix1 = ibox[0];
        auto iy1 = ibox[1];
        auto ix2 = ibox[2];
        auto iy2 = ibox[3];
        auto iw = ix2 - ix1;
        auto ih = iy2 - iy1;
        auto iarea = iw * ih;
        for (auto j = 0; j < n; j++) {
            auto jbox = boxes2 + 4 * j;
            auto jx1 = jbox[0];
            auto jy1 = jbox[1];
            auto jx2 = jbox[2];
            auto jy2 = jbox[3];
            auto jw = jx2 - jx1;
            auto jh = jy2 - jy1;
            auto jarea = jw * jh;

            auto xx1 = std::max(ix1, jx1);
            auto yy1 = std::max(iy1, jy1);
            auto xx2 = std::min(ix2, jx2);
            auto yy2 = std::min(iy2, jy2);

            auto w = std::max(static_cast<T>(0.0), xx2 - xx1);
            auto h = std::max(static_cast<T>(0.0), yy2 - yy1);
            auto inter = w * h;
            auto iou = inter / (iarea + jarea - inter);
            ious[i * n + j] = iou;
        }
    }
}

template <typename T>
void iou_mn_backward_kernel(T *dboxes1, T *dboxes2, const T *dious,
                            const T *boxes1, const T *boxes2, int64_t m,
                            int64_t n, const T *ious) {

    for (auto i = 0; i < m; i++) {
        auto ibox = boxes1 + 4 * i;
        auto dibox = dboxes1 + 4 * i;
        auto ix1 = ibox[0];
        auto iy1 = ibox[1];
        auto ix2 = ibox[2];
        auto iy2 = ibox[3];
        auto iw = ix2 - ix1;
        auto ih = iy2 - iy1;
        auto iarea = iw * ih;
        for (auto j = 0; j < n; j++) {
            if (ious[i * n + j] == 0)
                continue;
            auto jbox = boxes2 + 4 * j;
            auto djbox = dboxes2 + 4 * j;
            auto jx1 = jbox[0];
            auto jy1 = jbox[1];
            auto jx2 = jbox[2];
            auto jy2 = jbox[3];
            auto jw = jx2 - jx1;
            auto jh = jy2 - jy1;
            auto jarea = jw * jh;

            auto xx1 = std::max(ix1, jx1);
            auto yy1 = std::max(iy1, jy1);
            auto xx2 = std::min(ix2, jx2);
            auto yy2 = std::min(iy2, jy2);

            auto w = std::max(static_cast<T>(0.0), xx2 - xx1);
            auto h = std::max(static_cast<T>(0.0), yy2 - yy1);
            auto inter_area = w * h;
            auto union_area = iarea + jarea - inter_area;

            auto diou = dious[i * n + j];
            auto darea = diou * inter_area / (union_area * union_area);

            dibox[0] += ih * darea;
            dibox[1] += iw * darea;
            dibox[2] -= ih * darea;
            dibox[3] -= iw * darea;

            djbox[0] += jh * darea;
            djbox[1] += jw * darea;
            djbox[2] -= jh * darea;
            djbox[3] -= jw * darea;

            auto dinter =
                diou * (inter_area + union_area) / (union_area * union_area);
            auto dw = h * dinter;
            auto dh = w * dinter;

            if (ix1 >= jx1) {
                dibox[0] -= dw;
            } else {
                djbox[0] -= dw;
            }

            if (iy1 >= jy1) {
                dibox[1] -= dh;
            } else {
                djbox[1] -= dh;
            }

            if (ix2 <= jx2) {
                dibox[2] += dw;
            } else {
                djbox[2] += dw;
            }

            if (iy2 <= jy2) {
                dibox[3] += dh;
            } else {
                djbox[3] += dh;
            }
        }
    }
}

at::Tensor iou_mn_forward_cpu(const at::Tensor &boxes1,
                              const at::Tensor &boxes2) {
    AT_ASSERTM(!boxes1.type().is_cuda(), "boxes1 must be a CPU tensor");
    AT_ASSERTM(!boxes2.type().is_cuda(), "boxes2 must be a CPU tensor");
    AT_ASSERTM(boxes1.type() == boxes2.type(),
               "boxes1 should have the same type as boxes2");

    auto m = boxes1.size(0);
    auto n = boxes2.size(0);
    auto ious = torch::zeros({m, n}, boxes1.type());

    AT_DISPATCH_FLOATING_TYPES(boxes1.type(), "iou_mn_cpu", [&] {
        iou_mn_forward_kernel<scalar_t>(boxes1.contiguous().data<scalar_t>(),
                                        boxes2.contiguous().data<scalar_t>(), m,
                                        n, ious.data<scalar_t>());
    });
    return ious;
}

std::tuple<at::Tensor, at::Tensor> iou_mn_backward_cpu(const at::Tensor &dious,
                                                       const at::Tensor &boxes1,
                                                       const at::Tensor &boxes2,
                                                       const at::Tensor &ious) {
    AT_ASSERTM(!dious.type().is_cuda(), "dious must be a CPU tensor");
    AT_ASSERTM(!boxes1.type().is_cuda(), "boxes1 must be a CPU tensor");
    AT_ASSERTM(!boxes2.type().is_cuda(), "boxes2 must be a CPU tensor");

    // at::TensorArg dious_t{dious, "dious", 1}, boxes1_t{boxes1, "boxes1", 2},
    //     boxes2_t{boxes2, "boxes2", 3}, ious_t{ious, "ious", 4};

    // at::CheckedFrom c = "iou_mn_backward_cpu";
    // at::checkAllSameType(c, {dious_t, boxes1_t, boxes2_t, ious_t});

    auto m = boxes1.size(0);
    auto n = boxes2.size(0);
    at::Tensor dboxes1 = torch::zeros({m, 4}, boxes1.type());
    at::Tensor dboxes2 = torch::zeros({n, 4}, boxes2.type());

    if (dious.numel() == 0) {
        return std::make_tuple(dboxes1, dboxes2);
    }

    AT_DISPATCH_FLOATING_TYPES(boxes1.type(), "iou_mn_backward_cpu", [&] {
        iou_mn_backward_kernel<scalar_t>(
            dboxes1.data<scalar_t>(), dboxes2.data<scalar_t>(),
            dious.contiguous().data<scalar_t>(),
            boxes1.contiguous().data<scalar_t>(),
            boxes2.contiguous().data<scalar_t>(), m, n,
            ious.contiguous().data<scalar_t>());
    });
    return std::make_tuple(dboxes1, dboxes2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_cpu", &nms_cpu, "nms_cpu");
    m.def("soft_nms_cpu", &soft_nms_cpu, "soft_nms_cpu");
    m.def("iou_mn_forward_cpu", &iou_mn_forward_cpu, "iou_mn_forward_cpu");
    m.def("iou_mn_backward_cpu", &iou_mn_backward_cpu, "iou_mn_backward_cpu");
    m.def("psroi_align_forward", &PSROIAlign_forward, "PSROIAlign_forward");
    m.def("psroi_align_backward", &PSROIAlign_backward, "PSROIAlign_backward");
}