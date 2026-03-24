// ============================================================
// 自动生成的文件 — 请勿手动修改!
// 由 codegen.py 从 native_functions.yaml 生成
// ============================================================
#pragma once
// Tensor 方法实现 (调用 native 命名空间的 kernel)
inline Tensor Tensor::add(const Tensor& other) const { return native::add(*this, other); }
inline Tensor Tensor::mul(const Tensor& other) const { return native::mul(*this, other); }
inline Tensor Tensor::matmul(const Tensor& other) const { return native::matmul(*this, other); }
inline Tensor Tensor::relu() const { return native::relu(*this); }
inline float Tensor::sum() const { return native::sum(*this); }
inline Tensor Tensor::contiguous() const { return native::contiguous(*this); }
inline Tensor Tensor::transpose(int dim0, int dim1) const { return native::transpose(*this, dim0, dim1); }
inline Tensor Tensor::slice(int dim, int start, int end) const { return native::slice(*this, dim, start, end); }
inline Tensor Tensor::reshape(std::vector<int> shape) const { return native::reshape(*this, shape); }
inline Tensor Tensor::expand(std::vector<int> sizes) const { return native::expand(*this, sizes); }
