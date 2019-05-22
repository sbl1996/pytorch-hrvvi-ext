#include <torch/extension.h>

template <typename T>
std::vector<int64_t>
softer_nms_cpu_main(T *dets, T *scores, const T *vars, uint8_t *suppressed,
                    const T *areas, const int64_t ndets,
                    const float iou_threshold, const int topk,
                    const float sigma, const float min_score) {
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

        T *ibox = dets + 4 * i;
        auto ix1 = ibox[0];
        auto iy1 = ibox[1];
        auto ix2 = ibox[2];
        auto iy2 = ibox[3];

        const T *ivar = vars + 4 * i;
        auto ivx1 = ivar[0];
        auto ivy1 = ivar[1];
        auto ivx2 = ivar[2];
        auto ivy2 = ivar[3];

        auto px1_n = ix1 / ivx1;
        auto px1_d = 1 / ivx1;
        auto py1_n = iy1 / ivy1;
        auto py1_d = 1 / ivy1;
        auto px2_n = ix2 / ivx2;
        auto px2_d = 1 / ivx2;
        auto py2_n = iy2 / ivy2;
        auto py2_d = 1 / ivy2;
        for (int64_t j = 0; j < ndets; j++) {
            if (suppressed[j] == 1)
                continue;
            T *jbox = dets + 4 * j;
            auto jx1 = jbox[0];
            auto jy1 = jbox[1];
            auto jx2 = jbox[2];
            auto jy2 = jbox[3];

            const T *jvar = vars + 4 * j;
            auto jvx1 = jvar[0];
            auto jvy1 = jvar[1];
            auto jvx2 = jvar[2];
            auto jvy2 = jvar[3];

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
            if (iou > 0) {
                auto p = exp(-pow(1 - iou, 2) / sigma);
                px1_n += p * jx1 / jvx1;
                px1_d += p / jvx1;
                py1_n += p * jy1 / jvy1;
                py1_d += p / jvy1;
                px2_n += p * jx2 / jvx2;
                px2_d += p / jvx2;
                py2_n += p * jy2 / jvy2;
                py2_d += p / jvy2;
            }
        }
        ibox[0] = px1_n / px1_d;
        ibox[1] = py1_n / py1_d;
        ibox[2] = px2_n / px2_d;
        ibox[3] = py2_n / py2_d;
    }
    return indices;
}

template <typename scalar_t>
at::Tensor softer_nms_cpu_kernel(at::Tensor &dets_t, at::Tensor &scores_t,
                                 const at::Tensor &vars_t,
                                 const float iou_threshold, const int topk,
                                 const float sigma, const float min_score) {
    AT_ASSERTM(dets_t.is_contiguous(), "dets_t must be contiguous");
    AT_ASSERTM(!dets_t.type().is_cuda(), "dets_t must be a CPU tensor");
    AT_ASSERTM(!scores_t.type().is_cuda(), "scores_t must be a CPU tensor");
    AT_ASSERTM(!vars_t.type().is_cuda(), "vars_t must be a CPU tensor");
    AT_ASSERTM(dets_t.type() == scores_t.type(),
               "dets_t should have the same type as scores");
    AT_ASSERTM(scores_t.type() == vars_t.type(),
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
    auto scores = scores_t.data<scalar_t>();
    auto vars = vars_t.data<scalar_t>();
    auto suppressed = suppressed_t.data<uint8_t>();
    auto areas = areas_t.data<scalar_t>();
    std::vector<int64_t> indices =
        softer_nms_cpu_main(dets, scores, vars, suppressed, areas, ndets,
                            iou_threshold, topk, sigma, min_score);
    auto n = static_cast<int64_t>(indices.size());
    at::Tensor indices_t =
        torch::empty({n}, at::device(at::kCPU).dtype(at::kLong));
    auto indices_p = indices_t.data<int64_t>();
    for (auto i = 0; i < n; i++)
        indices_p[i] = indices[i];
    return indices_t;
}

at::Tensor softer_nms_cpu(at::Tensor &dets, at::Tensor &scores,
                          const at::Tensor &vars, const float iou_threshold,
                          const int topk, const float sigma,
                          const float min_score) {
    auto result = torch::empty({0}, dets.type());

    AT_DISPATCH_FLOATING_TYPES(dets.type(), "softer_nms_cpu", [&] {
        result = softer_nms_cpu_kernel<scalar_t>(
            dets, scores, vars, iou_threshold, topk, sigma, min_score);
    });
    return result;
}