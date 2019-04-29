#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "cuda_helpers.h"

#define GET_BLOCKS(block_size, n)                                              \
    ((static_cast<int>(n) + block_size - 1) / block_size)

template <typename T> __device__ inline float iou_11(const T *a, const T *b) {
    T left = max(a[0], b[0]), right = min(a[2], b[2]);
    T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
    T interS = width * height;
    T Sa = (a[2] - a[0]) * (a[3] - a[1]);
    T Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS / (Sa + Sb - interS);
}

template <typename T>
__device__ inline void iou_11_backward(T *dbox1, T *dbox2, const T dout,
                                       const T *box1, const T *box2,
                                       const T out) {
    if (out == 0) {
        return;
    }

    T ix1 = box1[0];
    T iy1 = box1[1];
    T ix2 = box1[2];
    T iy2 = box1[3];
    T iw = ix2 - ix1;
    T ih = iy2 - iy1;
    T iarea = iw * ih;

    T jx1 = box2[0];
    T jy1 = box2[1];
    T jx2 = box2[2];
    T jy2 = box2[3];
    T jw = jx2 - jx1;
    T jh = jy2 - jy1;
    T jarea = jw * jh;

    T xx1 = std::max(ix1, jx1);
    T yy1 = std::max(iy1, jy1);
    T xx2 = std::min(ix2, jx2);
    T yy2 = std::min(iy2, jy2);
    T w = std::max(static_cast<T>(0.0), xx2 - xx1);
    T h = std::max(static_cast<T>(0.0), yy2 - yy1);
    T inter_area = w * h;
    T union_area = iarea + jarea - inter_area;

    T darea = dout * inter_area / (union_area * union_area);

    atomicAdd(dbox1, ih * darea);
    atomicAdd(dbox1 + 1, iw * darea);
    atomicAdd(dbox1 + 2, -ih * darea);
    atomicAdd(dbox1 + 3, -iw * darea);

    atomicAdd(dbox2, jh * darea);
    atomicAdd(dbox2 + 1, jw * darea);
    atomicAdd(dbox2 + 2, -jh * darea);
    atomicAdd(dbox2 + 3, -jw * darea);

    T dinter = dout * (inter_area + union_area) / (union_area * union_area);
    T dw = h * dinter;
    T dh = w * dinter;

    if (ix1 >= jx1) {
        atomicAdd(dbox1, -dw);
    } else {
        atomicAdd(dbox2, -dw);
    }

    if (iy1 >= jy1) {
        atomicAdd(dbox1 + 1, -dh);
    } else {
        atomicAdd(dbox2 + 1, -dh);
    }

    if (ix2 <= jx2) {
        atomicAdd(dbox1 + 2, dw);
    } else {
        atomicAdd(dbox2 + 2, dw);
    }

    if (iy2 <= jy2) {
        atomicAdd(dbox1 + 3, dh);
    } else {
        atomicAdd(dbox2 + 3, dh);
    }
}

template <typename T>
__global__ void iou_mn_forward(const int nthreads, const T *boxes1,
                               const T *boxes2, const int m, const int n,
                               T *ious) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int j = index % n;
        int i = index / n;
        T iou = iou_11(boxes1 + i * 4, boxes2 + j * 4);
        ious[i * m + j] = iou;
    }
}

template <typename T>
__global__ void iou_mn_backward(const int nthreads, T *dboxes1, T *dboxes2,
                                const T *dout, const T *boxes1, const T *boxes2,
                                const int m, const int n, const T *ious) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int j = index % n;
        int i = index / n;
        iou_11_backward(dboxes1 + i * 4, dboxes2 + j * 4, dout[i * m + j],
                        boxes1 + i * 4, boxes2 + j * 4, ious[i * m + j]);
    }
}

at::Tensor iou_mn_forward_cuda(const at::Tensor &boxes1,
                               const at::Tensor &boxes2) {
    AT_ASSERTM(boxes1.device().is_cuda(), "boxes1 must be a CUDA tensor");
    AT_ASSERTM(boxes2.device().is_cuda(), "boxes2 must be a CUDA tensor");

    // at::TensorArg boxes1_t{boxes1, "boxes1", 1}, boxes2_t{boxes2, "boxes2",
    // 2};

    // at::CheckedFrom c = "iou_mn_forward_cuda";
    // at::checkAllSameGPU(c, {boxes1_t, boxes2_t});
    // at::checkAllSameType(c, {boxes1_t, boxes2_t});

    // at::cuda::CUDAGuard device_guard(boxes1.device());

    auto m = boxes1.size(0);
    auto n = boxes2.size(0);

    at::Tensor ious = at::zeros({m, n}, boxes1.options());

    auto output_size = m * n;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(GET_BLOCKS(512, output_size), 4096));
    dim3 block(512);

    if (ious.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return ious;
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        boxes1.type(), "iou_mn_forward_cuda", [&] {
            iou_mn_forward<scalar_t><<<grid, block, 0, stream>>>(
                output_size, boxes1.contiguous().data<scalar_t>(),
                boxes2.contiguous().data<scalar_t>(), m, n,
                ious.data<scalar_t>());
        });
    THCudaCheck(cudaGetLastError());
    return ious;
}

std::tuple<at::Tensor, at::Tensor>
iou_mn_backward_cuda(const at::Tensor &dout, const at::Tensor &boxes1,
                     const at::Tensor &boxes2, const at::Tensor &ious) {
    // Check if input tensors are CUDA tensors
    AT_ASSERTM(dout.device().is_cuda(), "dout must be a CUDA tensor");
    AT_ASSERTM(boxes1.device().is_cuda(), "boxes1 must be a CUDA tensor");
    AT_ASSERTM(boxes2.device().is_cuda(), "boxes2 must be a CUDA tensor");
    AT_ASSERTM(ious.device().is_cuda(), "ious must be a CUDA tensor");

    // at::TensorArg dout_t{dout, "dout", 1}, boxes1_t{boxes1, "boxes1", 2},
    //     boxes2_t{boxes2, "boxes2", 3}, ious_t{ious, "ious", 4};

    // at::CheckedFrom c = "iou_mn_backward_cuda";
    // at::checkAllSameGPU(c, {dout_t, boxes1_t, boxes2_t, ious_t});
    // at::checkAllSameType(c, {dout_t, boxes1_t, boxes2_t, ious_t});

    // at::cuda::CUDAGuard device_guard(dout.device());

    auto m = boxes1.size(0);
    auto n = boxes2.size(0);

    at::Tensor dboxes1 = at::zeros({m, 4}, boxes1.options());
    at::Tensor dboxes2 = at::zeros({n, 4}, boxes2.options());

    auto output_size = m * n;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(std::min(GET_BLOCKS(512, output_size), 4096));
    dim3 block(512);

    if (dout.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return std::make_tuple(dboxes1, dboxes2);
    }

    int m_stride = dout.stride(0);
    int n_stride = dout.stride(1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        dout.type(), "iou_mn_backward_cuda", [&] {
            iou_mn_backward<scalar_t><<<grid, block, 0, stream>>>(
                output_size, dboxes1.data<scalar_t>(), dboxes2.data<scalar_t>(),
                dout.contiguous().data<scalar_t>(),
                boxes1.contiguous().data<scalar_t>(),
                boxes2.contiguous().data<scalar_t>(),
                m, n,
                ious.contiguous().data<scalar_t>());
        });
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(dboxes1, dboxes2);
}
