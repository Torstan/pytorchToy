#pragma once

#include "../tensor.h"
#include "../ops.h"
#include "math.h"
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

    // 2D x 2D: 使用现有 matmul
    if (a_dim == 2 && b_dim == 2) {
        return native::matmul(a, b);
    }

    auto a_sizes = std::vector<int>(a.sizes());
    auto b_sizes = std::vector<int>(b.sizes());

    int M = a_sizes[a_dim - 2];
    int K = a_sizes[a_dim - 1];
    int N = b_sizes[b_dim - 1];
    if (K != b_sizes[b_dim - 2])
        throw std::runtime_error("batched_matmul: inner dimensions mismatch");

    // 提取 batch 维度并广播
    std::vector<int> a_batch(a_sizes.begin(), a_sizes.end() - 2);
    std::vector<int> b_batch(b_sizes.begin(), b_sizes.end() - 2);

    // 处理缺少 batch 维度的情况
    if (a_batch.empty()) a_batch.push_back(1);
    if (b_batch.empty()) b_batch.push_back(1);

    std::vector<int> batch_shape = broadcast_shape(a_batch, b_batch);
    int batch_size = numel_from_shape(batch_shape);

    // 结果形状
    std::vector<int> result_shape = batch_shape;
    result_shape.push_back(M);
    result_shape.push_back(N);

    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();

    // 确保输入连续
    Tensor ca = a.is_contiguous() ? Tensor(a) : native::contiguous(a);
    Tensor cb = b.is_contiguous() ? Tensor(b) : native::contiguous(b);
    const float* pa = ca.data_ptr();
    const float* pb = cb.data_ptr();

    auto a_batch_strides = contiguous_strides(a_batch);
    auto b_batch_strides = contiguous_strides(b_batch);

    int a_mat_size = M * K;
    int b_mat_size = K * N;
    int r_mat_size = M * N;

    for (int batch = 0; batch < batch_size; batch++) {
        auto coords = flat_to_coords(batch, batch_shape);

        // 计算 a, b 的 batch 偏移
        int a_offset = broadcast_flat_idx(coords, a_batch, a_batch_strides, batch_shape.size()) * a_mat_size;
        int b_offset = broadcast_flat_idx(coords, b_batch, b_batch_strides, batch_shape.size()) * b_mat_size;
        int r_offset = batch * r_mat_size;

        // 矩阵乘法内核
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float s = 0;
                for (int k = 0; k < K; k++)
                    s += pa[a_offset + i * K + k] * pb[b_offset + k * N + j];
                pr[r_offset + i * N + j] = s;
            }
        }
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
            // 找最大值
            float max_val = src[base];
            for (int d = 1; d < n_dim; d++) {
                float v = src[base + d * n_inner];
                if (v > max_val) max_val = v;
            }
            // exp 和 sum
            float sum = 0;
            for (int d = 0; d < n_dim; d++) {
                float e = std::exp(src[base + d * n_inner] - max_val);
                dst[base + d * n_inner] = e;
                sum += e;
            }
            // 归一化
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
