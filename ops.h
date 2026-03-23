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

// ============================================================
// Stride 感知的元素访问辅助
// ============================================================

// 将逻辑平坦索引 (0, 1, 2, ...) 转换为 storage 中的实际偏移
// 通过逐维度 "取模+除法" 将平坦索引分解为多维坐标，再用 strides 映射
inline int flat_to_storage_offset(int flat_idx, const std::vector<int>& sizes,
                                   const std::vector<int>& strides, int storage_offset) {
    int offset = storage_offset;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; i--) {
        int coord = flat_idx % sizes[i];
        flat_idx /= sizes[i];
        offset += coord * strides[i];
    }
    return offset;
}

// 从 tensor 中按逻辑索引读取元素（stride 感知）
inline float read_elem(const Tensor& t, int flat_idx) {
    if (t.is_contiguous()) return t.data_ptr()[flat_idx];
    int offset = flat_to_storage_offset(flat_idx, t.sizes(), t.strides(),
                                         t.storage_offset());
    return t.storage()->data[offset];
}

// 工厂函数（模拟 at::ones, at::empty 等）
inline Tensor ones(std::vector<int> shape) {
    return Tensor(TensorImplPtr(new TensorImpl(shape, 1.0f)));
}

inline Tensor fill(std::vector<int> shape, float value) {
    return Tensor(TensorImplPtr(new TensorImpl(shape, value)));
}

inline Tensor empty(std::vector<int> shape) {
    return Tensor(TensorImplPtr(new TensorImpl(shape, 0.0f)));
}

// ============================================================
// View 操作（零拷贝，共享 Storage）
// ============================================================

// contiguous — 如果已经连续直接返回自身，否则拷贝到新的连续 storage
inline Tensor contiguous(const Tensor& self) {
    if (self.is_contiguous()) return Tensor(self);
    Tensor result = empty(std::vector<int>(self.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < self.numel(); i++)
        pr[i] = read_elem(self, i);
    return result;
}

// transpose — 交换两个维度的 size 和 stride（零拷贝）
inline Tensor transpose(const Tensor& self, int dim0, int dim1) {
    if (dim0 < 0 || dim0 >= self.dim() || dim1 < 0 || dim1 >= self.dim())
        throw std::runtime_error("transpose: dimension out of range");
    std::vector<int> new_sizes(self.sizes());
    std::vector<int> new_strides(self.strides());
    std::swap(new_sizes[dim0], new_sizes[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    auto* impl = new TensorImpl(self.storage(), self.storage_offset(),
                                 std::move(new_sizes), std::move(new_strides));
    return Tensor(TensorImplPtr(impl));
}

// slice — 在指定维度取 [start, end) 子区间（零拷贝）
inline Tensor slice(const Tensor& self, int dim, int start, int end) {
    if (dim < 0 || dim >= self.dim())
        throw std::runtime_error("slice: dimension out of range");
    if (start < 0) start += self.sizes()[dim];
    if (end < 0) end += self.sizes()[dim];
    if (start < 0 || end > self.sizes()[dim] || start >= end)
        throw std::runtime_error("slice: invalid range");
    std::vector<int> new_sizes(self.sizes());
    new_sizes[dim] = end - start;
    int new_offset = self.storage_offset() + start * self.strides()[dim];
    auto* impl = new TensorImpl(self.storage(), new_offset,
                                 std::move(new_sizes),
                                 std::vector<int>(self.strides()));
    return Tensor(TensorImplPtr(impl));
}

// reshape — 改变形状。如果 contiguous 则零拷贝 view，否则先拷贝
inline Tensor reshape(const Tensor& self, std::vector<int> new_shape) {
    // 处理 -1 维度推断
    int neg_idx = -1;
    int known = 1;
    for (int i = 0; i < static_cast<int>(new_shape.size()); i++) {
        if (new_shape[i] == -1) {
            if (neg_idx >= 0)
                throw std::runtime_error("reshape: only one -1 allowed");
            neg_idx = i;
        } else {
            known *= new_shape[i];
        }
    }
    if (neg_idx >= 0) {
        if (self.numel() % known != 0)
            throw std::runtime_error("reshape: cannot infer dimension");
        new_shape[neg_idx] = self.numel() / known;
    }

    int new_numel = 1;
    for (int s : new_shape) new_numel *= s;
    if (new_numel != self.numel())
        throw std::runtime_error("reshape: total elements mismatch");

    Tensor src = self.is_contiguous() ? Tensor(self) : contiguous(self);

    // 计算新的行主序 strides
    std::vector<int> new_strides(new_shape.size());
    int stride = 1;
    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; i--) {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }
    auto* impl = new TensorImpl(src.storage(), src.storage_offset(),
                                 std::move(new_shape), std::move(new_strides));
    return Tensor(TensorImplPtr(impl));
}

// expand — 将 size=1 的维度广播到更大尺寸（stride 设为 0，零拷贝）
inline Tensor expand(const Tensor& self, std::vector<int> new_sizes) {
    if (static_cast<int>(new_sizes.size()) != self.dim())
        throw std::runtime_error("expand: dimension count mismatch");
    std::vector<int> new_strides(self.strides());
    for (int i = 0; i < self.dim(); i++) {
        if (new_sizes[i] == self.sizes()[i]) continue;
        if (self.sizes()[i] != 1)
            throw std::runtime_error("expand: can only expand size-1 dimensions");
        new_strides[i] = 0; // stride=0 实现广播：该维度所有位置读同一行
    }
    auto* impl = new TensorImpl(self.storage(), self.storage_offset(),
                                 std::move(new_sizes), std::move(new_strides));
    return Tensor(TensorImplPtr(impl));
}

// ============================================================
// 算子实现（stride 感知）
// ============================================================

// at::add — 逐元素加法
inline Tensor add(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("Tensor size mismatch in add");
    Tensor result = empty(std::vector<int>(a.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < a.numel(); i++)
        pr[i] = read_elem(a, i) + read_elem(b, i);
    return result;
}

// at::mul — 逐元素乘法
inline Tensor mul(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::runtime_error("Tensor size mismatch in mul");
    Tensor result = empty(std::vector<int>(a.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < a.numel(); i++)
        pr[i] = read_elem(a, i) * read_elem(b, i);
    return result;
}

// at::matmul — 矩阵乘法 (2D)，stride 感知
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() != 2 || b.dim() != 2)
        throw std::runtime_error("matmul requires 2D tensors");
    int M = a.sizes()[0], K = a.sizes()[1], N = b.sizes()[1];
    if (K != b.sizes()[0])
        throw std::runtime_error("matmul shape mismatch");
    Tensor result = empty({M, N});
    const float* sa = a.storage()->data.data();
    const float* sb = b.storage()->data.data();
    float* pr = result.data_ptr();
    int a_off = a.storage_offset(), b_off = b.storage_offset();
    int a_s0 = a.strides()[0], a_s1 = a.strides()[1];
    int b_s0 = b.strides()[0], b_s1 = b.strides()[1];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += sa[a_off + i * a_s0 + k * a_s1]
                   * sb[b_off + k * b_s0 + j * b_s1];
            pr[i * N + j] = s;
        }
    return result;
}

// at::relu
inline Tensor relu(const Tensor& a) {
    Tensor result = empty(std::vector<int>(a.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < a.numel(); i++) {
        float v = read_elem(a, i);
        pr[i] = v > 0 ? v : 0;
    }
    return result;
}

// at::sum
inline float sum(const Tensor& a) {
    float s = 0;
    for (int i = 0; i < a.numel(); i++) s += read_elem(a, i);
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
