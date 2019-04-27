#include "cpu/vision.h"
#include <ATen/TensorUtils.h>

template <typename T>
void iou_mn_forward_kernel(const T *boxes1, const T *boxes2, const int m,
                           const int n, T *ious) {

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
                            const T *boxes1, const T *boxes2, const int m,
                            const int n, const T *ious) {

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

    at::TensorArg boxes1_t{boxes1, "boxes1", 1}, boxes2_t{boxes2, "boxes2", 2};

    at::CheckedFrom c = "iou_mn_forward_cpu";
    at::checkAllSameType(c, {boxes1_t, boxes2_t});

    auto m = boxes1.size(0);
    auto n = boxes2.size(0);
    auto ious = torch::zeros({m, n}, boxes1.type());

    if (ious.numel() == 0)
        return ious;

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

    at::TensorArg dious_t{dious, "dious", 1}, boxes1_t{boxes1, "boxes1", 2},
        boxes2_t{boxes2, "boxes2", 3}, ious_t{ious, "ious", 4};

    at::CheckedFrom c = "iou_mn_backward_cpu";
    at::checkAllSameType(c, {dious_t, boxes1_t, boxes2_t, ious_t});

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