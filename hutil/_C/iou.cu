
template <typename T>
__device__ inline void iou_11(T *dbox1, T *dbox2, const T dout, const T *box1,
                              const T *box2) {

    auto ix1 = box1[0];
    auto iy1 = box1[1];
    auto ix2 = box1[2];
    auto iy2 = box1[3];
    auto iw = ix2 - ix1;
    auto ih = iy2 - iy1;
    auto iarea = iw * ih;

    auto jx1 = box2[0];
    auto jy1 = box2[1];
    auto jx2 = box2[2];
    auto jy2 = box2[3];
    auto jw = jx2 - jx1;
    auto jh = jy2 - jy1;
    auto jarea = jw * jh;

    auto xx1 = std::max(ix1, jx1);
    auto yy1 = std::max(iy1, jy1);
    auto xx2 = std::min(ix2, jx2);
    auto yy2 = std::min(iy2, jy2);
    auto w = std::max(static_cast<scalar_t>(0.0), xx2 - xx1);
    auto h = std::max(static_cast<scalar_t>(0.0), yy2 - yy1);
    auto inter_area = w * h;
    auto union_area = iarea + jarea - inter_area;

    auto darea = dout * inter_area / (union_area * union_area);

    atomicAdd(dbox1, ih * darea);
    atomicAdd(dbox1 + 1, iw * darea);
    atomicAdd(dbox1 + 2, -ih * darea);
    atomicAdd(dbox1 + 3, -iw * darea);

    atomicAdd(dbox2, jh * darea);
    atomicAdd(dbox2 + 1, jw * darea);
    atomicAdd(dbox2 + 2, -jh * darea);
    atomicAdd(dbox2 + 3, -jw * darea);

    auto dinter = dout * (inter_area + union_area) / (union_area * union_area);

    if (ix1 >= jx1) {
        atomicAdd(dbox1, -h * dinter);
    } else {
        atomicAdd(dbox2, -h * dinter);
    }

    if (iy1 >= jy1) {
        atomicAdd(dbox1 + 1, -w * dinter);
    } else {
        atomicAdd(dbox2 + 1, -w * dinter);
    }

    if (ix2 <= jx2) {
        atomicAdd(dbox1 + 2, h * dinter);
    } else {
        atomicAdd(dbox2 + 2, h * dinter);
    }

    if (iy2 <= jy2) {
        atomicAdd(dbox1 + 3, w * dinter);
    } else {
        atomicAdd(dbox2 + 3, w * dinter);
    }
}