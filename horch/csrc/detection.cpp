#include "IoUMN.h"
#include "PSROIAlign.h"
#include "nms.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms", &nms, "nms");
    m.def("soft_nms", &soft_nms, "soft_nms");
    m.def("softer_nms", &softer_nms, "softer_nms");
    m.def("iou_mn_forward", &iou_mn_forward, "iou_mn_forward");
    m.def("iou_mn_backward", &iou_mn_backward, "iou_mn_backward");
    m.def("psroi_align_forward", &PSROIAlign_forward, "PSROIAlign_forward");
    m.def("psroi_align_backward", &PSROIAlign_backward, "PSROIAlign_backward");
}