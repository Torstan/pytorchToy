#pragma once

#include "../tensor.h"
#include "../ops.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

// ============================================================
// 基础数学工具
// 提供 broadcast、逐元素数学函数、argmax 等通用工具
// ============================================================

namespace util {

// ---- 广播形状计算 ----
// 按 numpy 广播规则计算两个形状广播后的结果形状
inline std::vector<int> broadcast_shape(const std::vector<int>& a,
                                         const std::vector<int>& b) {
    int ndim = std::max(a.size(), b.size());
    std::vector<int> result(ndim);
    for (int i = 0; i < ndim; i++) {
        int da = i < (int)a.size() ? a[a.size() - 1 - i] : 1;
        int db = i < (int)b.size() ? b[b.size() - 1 - i] : 1;
        if (da != db && da != 1 && db != 1)
            throw std::runtime_error("broadcast_shape: incompatible shapes");
        result[ndim - 1 - i] = std::max(da, db);
    }
    return result;
}

// 计算总元素数
inline int numel_from_shape(const std::vector<int>& shape) {
    int n = 1;
    for (int s : shape) n *= s;
    return n;
}

// 计算行主序 strides
inline std::vector<int> contiguous_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size());
    int stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// 将平坦索引转为多维坐标
inline std::vector<int> flat_to_coords(int flat_idx, const std::vector<int>& shape) {
    int ndim = shape.size();
    std::vector<int> coords(ndim);
    for (int i = ndim - 1; i >= 0; i--) {
        coords[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    return coords;
}

// 将多维坐标映射到广播源 tensor 的平坦索引
inline int broadcast_flat_idx(const std::vector<int>& coords,
                               const std::vector<int>& src_shape,
                               const std::vector<int>& src_strides,
                               int ndim_result) {
    int ndim_src = src_shape.size();
    int offset = 0;
    for (int i = 0; i < ndim_src; i++) {
        int ri = ndim_result - ndim_src + i;
        int coord = (src_shape[i] == 1) ? 0 : coords[ri];
        offset += coord * src_strides[i];
    }
    return offset;
}

// ---- 广播二元操作通用实现 ----
inline Tensor broadcast_binary_op(const Tensor& a, const Tensor& b,
                                   float (*op)(float, float)) {
    auto result_shape = broadcast_shape(
        std::vector<int>(a.sizes()), std::vector<int>(b.sizes()));
    int n = numel_from_shape(result_shape);
    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();

    auto a_strides = contiguous_strides(std::vector<int>(a.sizes()));
    auto b_strides = contiguous_strides(std::vector<int>(b.sizes()));

    for (int i = 0; i < n; i++) {
        auto coords = flat_to_coords(i, result_shape);
        int ai = broadcast_flat_idx(coords, std::vector<int>(a.sizes()),
                                     a_strides, result_shape.size());
        int bi = broadcast_flat_idx(coords, std::vector<int>(b.sizes()),
                                     b_strides, result_shape.size());
        float va = a.unsafeGetTensorImpl()->read_logical(ai);
        float vb = b.unsafeGetTensorImpl()->read_logical(bi);
        pr[i] = op(va, vb);
    }
    return result;
}

// ---- 广播加法/乘法 ----
inline float _add_op(float a, float b) { return a + b; }
inline float _mul_op(float a, float b) { return a * b; }
inline float _sub_op(float a, float b) { return a - b; }
inline float _div_op(float a, float b) { return a / b; }

inline Tensor broadcast_add(const Tensor& a, const Tensor& b) {
    return broadcast_binary_op(a, b, _add_op);
}

inline Tensor broadcast_mul(const Tensor& a, const Tensor& b) {
    return broadcast_binary_op(a, b, _mul_op);
}

inline Tensor broadcast_sub(const Tensor& a, const Tensor& b) {
    return broadcast_binary_op(a, b, _sub_op);
}

inline Tensor broadcast_div(const Tensor& a, const Tensor& b) {
    return broadcast_binary_op(a, b, _div_op);
}

// ---- 逐元素数学函数 ----
inline Tensor elementwise_tanh(const Tensor& t) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        pr[i] = std::tanh(native::read_elem(t, i));
    return result;
}

inline Tensor elementwise_exp(const Tensor& t) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        pr[i] = std::exp(native::read_elem(t, i));
    return result;
}

inline Tensor elementwise_log(const Tensor& t) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        pr[i] = std::log(native::read_elem(t, i));
    return result;
}

inline Tensor elementwise_sqrt(const Tensor& t) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        pr[i] = std::sqrt(native::read_elem(t, i));
    return result;
}

inline Tensor elementwise_relu(const Tensor& t) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++) {
        float v = native::read_elem(t, i);
        pr[i] = v > 0 ? v : 0;
    }
    return result;
}

// 标量乘法
inline Tensor scalar_mul(const Tensor& t, float s) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        pr[i] = native::read_elem(t, i) * s;
    return result;
}

// 标量加法
inline Tensor scalar_add(const Tensor& t, float s) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        pr[i] = native::read_elem(t, i) + s;
    return result;
}

// ---- argmax ----
// 沿指定维度求最大值索引，返回整数值张量（用 float 存储）
inline Tensor argmax(const Tensor& t, int dim) {
    auto sizes = std::vector<int>(t.sizes());
    int ndim = sizes.size();
    if (dim < 0) dim += ndim;

    // 结果形状: 去掉 dim 维度
    std::vector<int> result_shape;
    for (int i = 0; i < ndim; i++)
        if (i != dim) result_shape.push_back(sizes[i]);
    if (result_shape.empty()) result_shape.push_back(1);

    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();

    // 计算外层/内层大小
    int n_outer = 1, n_inner = 1;
    for (int i = 0; i < dim; i++) n_outer *= sizes[i];
    for (int i = dim + 1; i < ndim; i++) n_inner *= sizes[i];
    int n_dim = sizes[dim];

    // 确保连续
    Tensor ct = t.is_contiguous() ? Tensor(t) : native::contiguous(t);
    const float* data = ct.data_ptr();
    int dim_stride = n_inner;
    int outer_stride = n_dim * n_inner;

    for (int outer = 0; outer < n_outer; outer++) {
        for (int inner = 0; inner < n_inner; inner++) {
            int base = outer * outer_stride + inner;
            float max_val = data[base];
            int max_idx = 0;
            for (int d = 1; d < n_dim; d++) {
                float v = data[base + d * dim_stride];
                if (v > max_val) {
                    max_val = v;
                    max_idx = d;
                }
            }
            pr[outer * n_inner + inner] = static_cast<float>(max_idx);
        }
    }
    return result;
}

// ---- 沿维度求和 ----
inline Tensor sum_dim(const Tensor& t, int dim, bool keepdim = false) {
    auto sizes = std::vector<int>(t.sizes());
    int ndim = sizes.size();
    if (dim < 0) dim += ndim;

    int n_outer = 1, n_inner = 1;
    for (int i = 0; i < dim; i++) n_outer *= sizes[i];
    for (int i = dim + 1; i < ndim; i++) n_inner *= sizes[i];
    int n_dim = sizes[dim];

    std::vector<int> result_shape;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            if (keepdim) result_shape.push_back(1);
        } else {
            result_shape.push_back(sizes[i]);
        }
    }
    if (result_shape.empty()) result_shape.push_back(1);

    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();

    Tensor ct = t.is_contiguous() ? Tensor(t) : native::contiguous(t);
    const float* data = ct.data_ptr();

    for (int outer = 0; outer < n_outer; outer++) {
        for (int inner = 0; inner < n_inner; inner++) {
            float s = 0;
            for (int d = 0; d < n_dim; d++)
                s += data[outer * n_dim * n_inner + d * n_inner + inner];
            pr[outer * n_inner + inner] = s;
        }
    }
    return result;
}

// 全局求和
inline float sum_all(const Tensor& t) {
    float s = 0;
    for (int i = 0; i < t.numel(); i++)
        s += native::read_elem(t, i);
    return s;
}

// ---- 沿维度求均值 ----
inline Tensor mean_dim(const Tensor& t, int dim, bool keepdim = false) {
    Tensor s = sum_dim(t, dim, keepdim);
    auto sizes = std::vector<int>(t.sizes());
    if (dim < 0) dim += (int)sizes.size();
    float n = (float)sizes[dim];
    float* pr = s.data_ptr();
    for (int i = 0; i < s.numel(); i++)
        pr[i] /= n;
    return s;
}

// ---- 沿维度求方差 ----
inline Tensor var_dim(const Tensor& t, int dim, bool keepdim = false) {
    auto sizes = std::vector<int>(t.sizes());
    int ndim = sizes.size();
    if (dim < 0) dim += ndim;

    Tensor mean = mean_dim(t, dim, true); // keepdim=true for broadcasting

    int n_outer = 1, n_inner = 1;
    for (int i = 0; i < dim; i++) n_outer *= sizes[i];
    for (int i = dim + 1; i < ndim; i++) n_inner *= sizes[i];
    int n_dim = sizes[dim];

    std::vector<int> result_shape;
    for (int i = 0; i < ndim; i++) {
        if (i == dim) {
            if (keepdim) result_shape.push_back(1);
        } else {
            result_shape.push_back(sizes[i]);
        }
    }
    if (result_shape.empty()) result_shape.push_back(1);

    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();
    const float* pm = mean.data_ptr();

    Tensor ct = t.is_contiguous() ? Tensor(t) : native::contiguous(t);
    const float* data = ct.data_ptr();

    for (int outer = 0; outer < n_outer; outer++) {
        for (int inner = 0; inner < n_inner; inner++) {
            float s = 0;
            float m = pm[outer * n_inner + inner];
            for (int d = 0; d < n_dim; d++) {
                float v = data[outer * n_dim * n_inner + d * n_inner + inner] - m;
                s += v * v;
            }
            pr[outer * n_inner + inner] = s / n_dim;
        }
    }
    return result;
}

// ---- 随机数填充 ----
inline void fill_randn(Tensor& t, unsigned seed = 0) {
    std::mt19937 gen(seed ? seed : std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* p = t.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        p[i] = dist(gen);
}

inline void fill_uniform(Tensor& t, float low, float high, unsigned seed = 0) {
    std::mt19937 gen(seed ? seed : std::random_device{}());
    std::uniform_real_distribution<float> dist(low, high);
    float* p = t.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        p[i] = dist(gen);
}

// ---- 创建随机整数张量 ----
inline Tensor randint(int low, int high, std::vector<int> shape, unsigned seed = 0) {
    std::mt19937 gen(seed ? seed : std::random_device{}());
    std::uniform_int_distribution<int> dist(low, high - 1);
    Tensor result = native::empty(shape);
    float* p = result.data_ptr();
    for (int i = 0; i < result.numel(); i++)
        p[i] = static_cast<float>(dist(gen));
    return result;
}

// ---- 拼接 ----
inline Tensor cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::runtime_error("cat: empty tensor list");

    auto base_sizes = std::vector<int>(tensors[0].sizes());
    int ndim = base_sizes.size();
    if (dim < 0) dim += ndim;

    // 计算拼接后的形状
    int cat_size = 0;
    for (auto& t : tensors) {
        auto s = std::vector<int>(t.sizes());
        cat_size += s[dim];
    }

    std::vector<int> result_shape = base_sizes;
    result_shape[dim] = cat_size;
    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();

    // 计算外层/内层
    int n_outer = 1, n_inner = 1;
    for (int i = 0; i < dim; i++) n_outer *= result_shape[i];
    for (int i = dim + 1; i < ndim; i++) n_inner *= result_shape[i];

    int offset = 0;
    for (auto& t : tensors) {
        auto s = std::vector<int>(t.sizes());
        int t_dim = s[dim];
        Tensor ct = t.is_contiguous() ? Tensor(t) : native::contiguous(t);
        const float* src = ct.data_ptr();

        for (int outer = 0; outer < n_outer; outer++) {
            for (int d = 0; d < t_dim; d++) {
                for (int inner = 0; inner < n_inner; inner++) {
                    int src_idx = outer * t_dim * n_inner + d * n_inner + inner;
                    int dst_idx = outer * cat_size * n_inner + (offset + d) * n_inner + inner;
                    pr[dst_idx] = src[src_idx];
                }
            }
        }
        offset += t_dim;
    }
    return result;
}

// ---- 切分 ----
inline std::vector<Tensor> chunk(const Tensor& t, int n, int dim) {
    auto sizes = std::vector<int>(t.sizes());
    if (dim < 0) dim += (int)sizes.size();
    int total = sizes[dim];
    int chunk_size = (total + n - 1) / n;

    std::vector<Tensor> result;
    for (int i = 0; i < n; i++) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start >= total) break;
        result.push_back(native::slice(t, dim, start, end));
    }
    return result;
}

// ---- 转置最后两维 ----
inline Tensor transpose_last2(const Tensor& t) {
    int ndim = t.dim();
    if (ndim < 2)
        throw std::runtime_error("transpose_last2: need at least 2D");
    return native::transpose(t, ndim - 2, ndim - 1);
}

// ---- 克隆(深拷贝) ----
inline Tensor clone(const Tensor& t) {
    Tensor result = native::empty(std::vector<int>(t.sizes()));
    float* pr = result.data_ptr();
    for (int i = 0; i < t.numel(); i++)
        pr[i] = native::read_elem(t, i);
    return result;
}

// ---- zeros_like / ones_like ----
inline Tensor zeros_like(const Tensor& t) {
    return native::empty(std::vector<int>(t.sizes()));
}

inline Tensor ones_like(const Tensor& t) {
    return native::ones(std::vector<int>(t.sizes()));
}

} // namespace util
