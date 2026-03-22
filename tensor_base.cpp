// ============================================================
// TensorBase 的独立编译单元
// 对应 PyTorch: aten/src/ATen/core/TensorBase.cpp
//
// 关键点: 此文件只依赖 tensor.h (TensorBase)，不依赖:
//   - native_functions.yaml
//   - ops.h
//   - generated/ 下的任何文件
//
// 因此修改算子定义时，此文件不需要重新编译。
// 在 PyTorch 这样的大型项目中，类似的文件有数百个，
// 避免重编译它们可以节省大量增量编译时间。
// ============================================================

#include "tensor_base.h"

std::string TensorBase::repr() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (int i = 0; i < dim(); i++) {
        if (i > 0) oss << ", ";
        oss << sizes()[i];
    }
    oss << "], data=[";
    int n = std::min(numel(), 6);
    const float* d = data_ptr();
    for (int i = 0; i < n; i++) {
        if (i > 0) oss << ", ";
        oss << d[i];
    }
    if (numel() > 6) oss << ", ...";
    oss << "])";
    return oss.str();
}
