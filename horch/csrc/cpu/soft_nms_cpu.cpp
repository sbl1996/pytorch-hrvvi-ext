#include <torch/extension.h>

template <typename T>
std::vector<int64_t>
soft_nms_cpu_main(const T *dets, T *scores, uint8_t *suppressed, const T *areas,
                  const int64_t ndets, const float iou_threshold,
                  const int topk, const float min_score) {
    std::vector<int64_t> indices;

    for (int64_t _i = 0; _i < topk; _i++) {
        T max_score = 0;
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

        const T *ibox = dets + 4 * i;
        auto ix1 = ibox[0];
        auto iy1 = ibox[1];
        auto ix2 = ibox[2];
        auto iy2 = ibox[3];

        for (int64_t j = 0; j < ndets; j++) {
            if (suppressed[j] == 1)
                continue;
            const T *jbox = dets + 4 * j;
            auto jx1 = jbox[0];
            auto jy1 = jbox[1];
            auto jx2 = jbox[2];
            auto jy2 = jbox[3];

            auto xx1 = std::max(ix1, jx1);
            auto yy1 = std::max(iy1, jy1);
            auto xx2 = std::min(ix2, jx2);
            auto yy2 = std::min(iy2, jy2);

            auto w =
                std::max(static_cast<T>(0), xx2 - xx1 + static_cast<T>(0.01));
            auto h =
                std::max(static_cast<T>(0), yy2 - yy1 + static_cast<T>(0.01));
            auto inter = w * h;
            auto iou = inter / (areas[i] + areas[j] - inter);
            if (iou >= iou_threshold) {
                scores[j] *= 1 - iou;
                if (scores[j] < min_score) {
                    suppressed[j] = 1;
                }
            }
        }
    }
    return indices;
}

template <typename scalar_t>
at::Tensor soft_nms_cpu_kernel(const at::Tensor &dets_t, at::Tensor &scores_t,
                               const float iou_threshold, const int topk,
                               const float min_score) {
    AT_ASSERTM(!dets_t.type().is_cuda(), "dets_t must be a CPU tensor");
    AT_ASSERTM(!scores_t.type().is_cuda(), "scores must be a CPU tensor");
    AT_ASSERTM(dets_t.type() == scores_t.type(),
               "dets_t should have the same type as scores");

    if (dets_t.numel() == 0)
        return torch::empty({0}, at::device(at::kCPU).dtype(at::kLong));

    auto x1_t = dets_t.select(1, 0);
    auto y1_t = dets_t.select(1, 1);
    auto x2_t = dets_t.select(1, 2);
    auto y2_t = dets_t.select(1, 3);
    at::Tensor areas_t = (x2_t - x1_t + 0.01) * (y2_t - y1_t + 0.01);

    auto ndets = dets_t.size(0);
    at::Tensor suppressed_t =
        torch::zeros({ndets}, at::device(at::kCPU).dtype(at::kByte));

    auto dets = dets_t.data<scalar_t>();
    auto suppressed = suppressed_t.data<uint8_t>();
    auto areas = areas_t.data<scalar_t>();
    auto scores = scores_t.data<scalar_t>();
    std::vector<int64_t> indices = soft_nms_cpu_main(
        dets, scores, suppressed, areas, ndets, iou_threshold, topk, min_score);

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
                        const float min_score) {

    auto result = torch::empty({0}, dets.type());

    AT_DISPATCH_FLOATING_TYPES(dets.type(), "soft_nms_cpu", [&] {
        result = soft_nms_cpu_kernel<scalar_t>(dets.contiguous(), scores,
                                               iou_threshold, topk, min_score);
    });
    return result;
}