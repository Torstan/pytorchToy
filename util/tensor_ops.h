#pragma once

#include "../tensor.h"
#include "../ops.h"
#include "math.h"
#include "gemm.h"
#include <vector>
#include <stdexcept>
#include <algorithm>

// ============================================================
// 高级张量操作
// batched_matmul、高维 reshape/transpose 等
// ============================================================

namespace util {

// ============================================================
// batched_matmul — 支持 2D/3D/4D 矩阵乘法，带广播
//
// 规则 (同 PyTorch torch.matmul):
// - 2D x 2D: 普通矩阵乘法 [M,K] x [K,N] → [M,N]
// - 3D x 3D: batch matmul [B,M,K] x [B,K,N] → [B,M,N]
// - 高维: 最后两维做矩阵乘，前面的维度做广播
// ============================================================

inline Tensor batched_matmul(const Tensor& a, const Tensor& b) {
    int a_dim = a.dim();
    int b_dim = b.dim();

    // 2D x 2D: 直接调用 i-k-j 内核
    if (a_dim == 2 && b_dim == 2) {
        auto a_sizes = std::vector<int>(a.sizes());
        auto b_sizes = std::vector<int>(b.sizes());
        int M = a_sizes[0], K = a_sizes[1], N = b_sizes[1];
        if (K != b_sizes[0])
            throw std::runtime_error("batched_matmul: inner dimensions mismatch");
        Tensor ca = a.is_contiguous() ? Tensor(a) : native::contiguous(a);
        Tensor cb = b.is_contiguous() ? Tensor(b) : native::contiguous(b);
        Tensor result = native::empty({M, N});
        gemm::matmul(ca.data_ptr(), cb.data_ptr(), result.data_ptr(), M, K, N);
        return result;
    }

    auto a_sizes = std::vector<int>(a.sizes());
    auto b_sizes = std::vector<int>(b.sizes());

    int M = a_sizes[a_dim - 2];
    int K = a_sizes[a_dim - 1];
    int N = b_sizes[b_dim - 1];
    if (K != b_sizes[b_dim - 2])
        throw std::runtime_error("batched_matmul: inner dimensions mismatch");

    Tensor ca = a.is_contiguous() ? Tensor(a) : native::contiguous(a);
    Tensor cb = b.is_contiguous() ? Tensor(b) : native::contiguous(b);
    const float* pa = ca.data_ptr();
    const float* pb = cb.data_ptr();

    int a_mat_size = M * K;
    int b_mat_size = K * N;
    int r_mat_size = M * N;

    // ---- 快速路径: 3D 无广播 [B,M,K] x [B,K,N] ----
    if (a_dim == 3 && b_dim == 3 && a_sizes[0] == b_sizes[0]) {
        int B = a_sizes[0];
        Tensor result = native::empty({B, M, N});
        float* pr = result.data_ptr();
        for (int batch = 0; batch < B; batch++)
            gemm::matmul(pa + batch * a_mat_size, pb + batch * b_mat_size,
                       pr + batch * r_mat_size, M, K, N);
        return result;
    }

    // ---- 快速路径: 4D 无广播 [B1,B2,M,K] x [B1,B2,K,N] ----
    if (a_dim == 4 && b_dim == 4 &&
        a_sizes[0] == b_sizes[0] && a_sizes[1] == b_sizes[1]) {
        int B1 = a_sizes[0], B2 = a_sizes[1];
        Tensor result = native::empty({B1, B2, M, N});
        float* pr = result.data_ptr();
        for (int b1 = 0; b1 < B1; b1++)
            for (int b2 = 0; b2 < B2; b2++) {
                int idx = b1 * B2 + b2;
                gemm::matmul(pa + idx * a_mat_size, pb + idx * b_mat_size,
                           pr + idx * r_mat_size, M, K, N);
            }
        return result;
    }

    // ---- 通用广播路径 ----
    std::vector<int> a_batch(a_sizes.begin(), a_sizes.end() - 2);
    std::vector<int> b_batch(b_sizes.begin(), b_sizes.end() - 2);
    if (a_batch.empty()) a_batch.push_back(1);
    if (b_batch.empty()) b_batch.push_back(1);

    std::vector<int> batch_shape = broadcast_shape(a_batch, b_batch);
    int batch_size = numel_from_shape(batch_shape);

    std::vector<int> result_shape = batch_shape;
    result_shape.push_back(M);
    result_shape.push_back(N);

    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();

    auto a_batch_strides = contiguous_strides(a_batch);
    auto b_batch_strides = contiguous_strides(b_batch);

    for (int batch = 0; batch < batch_size; batch++) {
        auto coords = flat_to_coords(batch, batch_shape);
        int a_offset = broadcast_flat_idx(coords, a_batch, a_batch_strides, batch_shape.size()) * a_mat_size;
        int b_offset = broadcast_flat_idx(coords, b_batch, b_batch_strides, batch_shape.size()) * b_mat_size;
        int r_offset = batch * r_mat_size;
        gemm::matmul(pa + a_offset, pb + b_offset, pr + r_offset, M, K, N);
    }
    return result;
}

// ============================================================
// softmax — 沿指定维度，数值稳定版本
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// ============================================================
inline Tensor softmax(const Tensor& t, int dim) {
    auto sizes = std::vector<int>(t.sizes());
    int ndim = sizes.size();
    if (dim < 0) dim += ndim;

    int n_outer = 1, n_inner = 1;
    for (int i = 0; i < dim; i++) n_outer *= sizes[i];
    for (int i = dim + 1; i < ndim; i++) n_inner *= sizes[i];
    int n_dim = sizes[dim];

    Tensor ct = t.is_contiguous() ? Tensor(t) : native::contiguous(t);
    Tensor result = native::empty(sizes);
    const float* src = ct.data_ptr();
    float* dst = result.data_ptr();

    for (int outer = 0; outer < n_outer; outer++) {
        for (int inner = 0; inner < n_inner; inner++) {
            int base = outer * n_dim * n_inner + inner;
            float max_val = src[base];
            for (int d = 1; d < n_dim; d++) {
                float v = src[base + d * n_inner];
                if (v > max_val) max_val = v;
            }
            float sum = 0;
            for (int d = 0; d < n_dim; d++) {
                float e = std::exp(src[base + d * n_inner] - max_val);
                dst[base + d * n_inner] = e;
                sum += e;
            }
            for (int d = 0; d < n_dim; d++) {
                dst[base + d * n_inner] /= sum;
            }
        }
    }
    return result;
}

// ============================================================
// log_softmax — 数值稳定版本
// log_softmax(x_i) = x_i - max - log(sum(exp(x_j - max)))
// ============================================================
inline Tensor log_softmax(const Tensor& t, int dim) {
    auto sizes = std::vector<int>(t.sizes());
    int ndim = sizes.size();
    if (dim < 0) dim += ndim;

    int n_outer = 1, n_inner = 1;
    for (int i = 0; i < dim; i++) n_outer *= sizes[i];
    for (int i = dim + 1; i < ndim; i++) n_inner *= sizes[i];
    int n_dim = sizes[dim];

    Tensor ct = t.is_contiguous() ? Tensor(t) : native::contiguous(t);
    Tensor result = native::empty(sizes);
    const float* src = ct.data_ptr();
    float* dst = result.data_ptr();

    for (int outer = 0; outer < n_outer; outer++) {
        for (int inner = 0; inner < n_inner; inner++) {
            int base = outer * n_dim * n_inner + inner;
            float max_val = src[base];
            for (int d = 1; d < n_dim; d++) {
                float v = src[base + d * n_inner];
                if (v > max_val) max_val = v;
            }
            float sum = 0;
            for (int d = 0; d < n_dim; d++)
                sum += std::exp(src[base + d * n_inner] - max_val);
            float log_sum = std::log(sum) + max_val;
            for (int d = 0; d < n_dim; d++)
                dst[base + d * n_inner] = src[base + d * n_inner] - log_sum;
        }
    }
    return result;
}

} // namespace util
