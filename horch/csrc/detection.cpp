#include "Deform.h"
#include "IoUMN.h"
#include "PSROIAlign.h"
#include "ROIAlign.h"
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
    m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
    m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
#ifdef WITH_CUDA
    m.def("deform_conv_forward_cuda", &deform_conv_forward_cuda,
          "deform forward (CUDA)");
    m.def("deform_conv_backward_input_cuda", &deform_conv_backward_input_cuda,
          "deform_conv_backward_input (CUDA)");
    m.def("deform_conv_backward_parameters_cuda",
          &deform_conv_backward_parameters_cuda,
          "deform_conv_backward_parameters (CUDA)");
    m.def("modulated_deform_conv_cuda_forward",
          &modulated_deform_conv_cuda_forward,
          "modulated deform conv forward (CUDA)");
    m.def("modulated_deform_conv_cuda_backward",
          &modulated_deform_conv_cuda_backward,
          "modulated deform conv backward (CUDA)");
#endif
}