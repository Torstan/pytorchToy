#pragma once
#include "tensor.h"

// ============================================================
// 模拟 at::native 算子层
// 对应 PyTorch: aten/src/ATen/native/*.cpp
//
// 这些是算子的真正 C++ 实现（kernel）。
// 在 PyTorch 中，Dispatcher 根据 DispatchKey 选择对应的 kernel。
// 这里简化为直接调用。
// ============================================================

namespace native {

// 工厂函数（模拟 at::ones, at::empty 等）
inline Tensor ones(std::vector<int> shape) {
    return Tensor(IntrusivePtr(new TensorImpl(shape, 1.0f)));
}

inline Tensor fill(std::vector<int> shape, float value) {
    return Tensor(IntrusivePtr(new TensorImpl(shape, value)));
}

inline Tensor empty(std::vector<int> shape) {
    return Tensor(IntrusivePtr(new TensorImpl(shape, 0.0f)));
}

// at::add — 逐元素加法
inline Tensor add(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("Tensor size mismatch in add");
    Tensor result = empty(std::vector<int>(a.sizes()));
    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    float* pr = result.data_ptr();
    for (int i = 0; i < a.numel(); i++)
        pr[i] = pa[i] + pb[i];
    return result;
}

// at::mul — 逐元素乘法
inline Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("Tensor size mismatch in mul");
    Tensor result = empty(std::vector<int>(a.sizes()));
    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    float* pr = result.data_ptr();
    for (int i = 0; i < a.numel(); i++)
        pr[i] = pa[i] * pb[i];
    return result;
}

// at::matmul — 矩阵乘法 (2D)
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2)
        throw std::runtime_error("matmul requires 2D tensors");
    int M = a.sizes()[0], K = a.sizes()[1], N = b.sizes()[1];
    if (K != b.sizes()[0])
        throw std::runtime_error("matmul shape mismatch");
    Tensor result = empty({M, N});
    const float* pa = a.data_ptr();
    const float* pb = b.data_ptr();
    float* pr = result.data_ptr();
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += pa[i * K + k] * pb[k * N + j];
            pr[i * N + j] = s;
        }
    return result;
}

// at::relu
inline Tensor relu(const Tensor& a) {
    Tensor result = empty(std::vector<int>(a.sizes()));
    const float* pa = a.data_ptr();
    float* pr = result.data_ptr();
    for (int i = 0; i < a.numel(); i++)
        pr[i] = pa[i] > 0 ? pa[i] : 0;
    return result;
}

// at::sum
inline float sum(const Tensor& a) {
    float s = 0;
    const float* p = a.data_ptr();
    for (int i = 0; i < a.numel(); i++) s += p[i];
    return s;
}

} // namespace native

// ============================================================
// Tensor 方法的实现
// 在 PyTorch 中，这些方法体由 torchgen 代码生成器自动生成，
// 内部通过 Dispatcher 分派到上面的 native 函数。
// 由 codegen.py 从 native_functions.yaml 生成:
// ============================================================

#include "generated/dispatch.h"
