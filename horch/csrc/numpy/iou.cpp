
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define loop(i, n) for (int i = 0; i < n; i++)

using namespace std;


template <typename T>
vector<T> index(const vector<T> &xs, const vector<int64_t> &indices) {
    int64_t n = xs.size();
    vector<T> ret(n);
    loop(i, n) { ret[i] = xs[indices[i]]; }
    return ret;
}

template <typename T> void print_vector(const vector<T> &xs) {
    cout << "[";
    int n = xs.size();
    if (n != 0) {
        cout << " " << xs[0];
        for (int i = 1; i < n; i++) {
            cout << ", " << xs[i];
        }
        cout << " ";
    }
    cout << "]";
}

template <typename T> inline T iou_11(const T *a, const T *b) {
    T left = max(a[0], b[0]), right = min(a[2], b[2]);
    T top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    T width = max(right - left, (T)0), height = max(bottom - top, (T)0);
    T interS = width * height;
    T Sa = (a[2] - a[0]) * (a[3] - a[1]);
    T Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS / (Sa + Sb - interS);
}

template <typename T> void non_max_suppression(const T *img, int64_t m, int64_t n, const float *angle, T *out) {

    for (int i = 1; i < m - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            auto q = 255;
            auto r = 255;
            auto c = i * n + j;
            auto a = angle[c];
            if (((0 <= a) && (a < 22.5)) || (157.5 <= a) && (a <= 180)) {
                q = img[i * n + j + 1];
                r = img[i * n + j - 1];
            }
            else if ((22.5 <= a) && (a < 67.5)) {
                q = img[(i + 1) * n + j - 1];
                r = img[(i - 1) * n + j + 1];
            }
            else if ((67.5 <= a) && (a < 112.5)) {
                q = img[(i + 1) * n + j];
                r = img[(i - 1) * n + j];
            }
            else if ((112.5 <= a) && (a < 157.5)) {
                q = img[(i - 1) * n + j - 1];
                r = img[(i + 1) * n + j + 1];
            }

            if ((img[c] >= q) && (img[c] >= r)) {
                out[c] = img[c];
            }
            else {
                out[c] = 0;
            }
        }
    }
}

template <typename T> void iou_mm(const T *boxes, int64_t n, T *out) {

    vector<T> areas(n);
    for (auto i = 0; i < n; i++) {
        auto box = boxes + 4 * i;
        auto w = box[2] - box[0];
        auto h = box[3] - box[1];
        areas[i] = w * h;
    }

    for (auto i = 0; i < n; i++) {
        auto ibox = boxes + 4 * i;
        auto ix1 = ibox[0];
        auto iy1 = ibox[1];
        auto ix2 = ibox[2];
        auto iy2 = ibox[3];
        auto iarea = areas[i];
        for (auto j = i + 1; j < n; j++) {
            auto jbox = boxes + 4 * j;
            auto xx1 = std::max(ix1, jbox[0]);
            auto yy1 = std::max(iy1, jbox[1]);
            auto xx2 = std::min(ix2, jbox[2]);
            auto yy2 = std::min(iy2, jbox[3]);

            auto w = std::max(static_cast<T>(0.0), xx2 - xx1);
            auto h = std::max(static_cast<T>(0.0), yy2 - yy1);
            auto inter = w * h;
            auto iou = inter / (iarea + areas[j] - inter);
            out[i * n + j] = iou;
        }
    }
    for (auto i = 0; i < n; i++) {
        out[i * n + i] = 1;
    }
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < i; j++) {
            out[i * n + j] = out[j * n + i];
        }
    }
}

template <typename T>
void iou_mn(const T *boxes1, int64_t m, const T *boxes2, int64_t n, T *out) {

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
            out[i * n + j] = iou;
        }
    }
}

template <typename T> double giou_11(const T *box1, const T *box2) {

    auto ix1 = box1[0];
    auto iy1 = box1[1];
    auto ix2 = box1[2];
    auto iy2 = box1[3];
    auto iarea = (ix2 - ix1) * (iy2 - iy1);

    auto jx1 = box2[0];
    auto jy1 = box2[1];
    auto jx2 = box2[2];
    auto jy2 = box2[3];
    auto jarea = (jx2 - jx1) * (jy2 - jy1);

    auto xx1 = std::max(ix1, jx1);
    auto yy1 = std::max(iy1, jy1);
    auto xx2 = std::min(ix2, jx2);
    auto yy2 = std::min(iy2, jy2);

    auto w = std::max(static_cast<T>(0.0), xx2 - xx1);
    auto h = std::max(static_cast<T>(0.0), yy2 - yy1);
    auto inter_area = w * h;
    auto union_area = iarea + jarea - inter_area;

    auto cx1 = std::min(ix1, jx1);
    auto cy1 = std::min(iy1, jy1);
    auto cx2 = std::max(ix2, jx2);
    auto cy2 = std::max(iy2, jy2);
    auto carea = (cx2 - cx1) * (cy2 - cy1);

    auto iou = inter_area / union_area;
    auto giou = iou - (carea - union_area) / carea;
    return giou;
}

template <typename T>
py::array_t<T>
Py_iou_mm(py::array_t<T, py::array::c_style | py::array::forcecast> boxes) {
    int64_t n = boxes.shape(0);
    auto out = py::array_t<T>({n, n});
    iou_mm(boxes.data(), n, out.mutable_data());
    return out;
}

template <typename T>
py::array_t<T>
Py_non_max_suppression(
    py::array_t<T, py::array::c_style | py::array::forcecast> img,
    py::array_t<float, py::array::c_style | py::array::forcecast> angle) {
    int64_t m = img.shape(0);
    int64_t n = img.shape(1);
    auto out = py::array_t<T>({m, n});
    non_max_suppression(img.data(), m, n, angle.data(), out.mutable_data());
    return out;
}

template <typename T>
py::array_t<T>
Py_iou_mn(py::array_t<T, py::array::c_style | py::array::forcecast> boxes1,
          py::array_t<T, py::array::c_style | py::array::forcecast> boxes2) {
    int64_t m = boxes1.shape(0);
    int64_t n = boxes2.shape(0);
    auto out = py::array_t<T>({m, n});
    iou_mn(boxes1.data(), m, boxes2.data(), n, out.mutable_data());
    return out;
}

template <typename T>
T Py_iou_11(py::array_t<T, py::array::c_style | py::array::forcecast> box1,
            py::array_t<T, py::array::c_style | py::array::forcecast> box2) {
    return iou_11(box1.data(), box2.data());
}

PYBIND11_MODULE(_numpy, m) {
    m.def("iou_mm", &Py_iou_mm<double>,
          "Calculate ious for boxes with themselves.");
    m.def("iou_mm", &Py_iou_mm<float>,
          "Calculate ious for boxes with themselves.");
    m.def("iou_mn", &Py_iou_mn<double>, "iou_mn");
    m.def("iou_mn", &Py_iou_mn<float>, "iou_mn");
    m.def("iou_11", &Py_iou_11<double>, "iou_11");
    m.def("iou_11", &Py_iou_11<float>, "iou_11");
    m.def("ed_nms", &Py_non_max_suppression<uint8_t>, "ed_nms");
    m.def("ed_nms", &Py_non_max_suppression<float>, "ed_nms");
    m.def("ed_nms", &Py_non_max_suppression<double>, "ed_nms");
}