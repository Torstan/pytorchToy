#pragma once

#include "function.h"
#include "../ops.h"
#include "../util/math.h"
#include "../util/tensor_ops.h"
#include "../nn/ops.h"

// ============================================================
// 内置算子的 backward 实现
// ============================================================

// ---- 辅助: reduce_grad ----
// 沿广播维度求和，将梯度形状还原为原始输入形状
// 对应 Python 侧的 _reduce_grad()
namespace autograd_util {

inline Tensor reduce_grad(const Tensor& grad, const std::vector<int>& target_shape) {
    const auto& grad_shape = grad.sizes();
    if (grad_shape == target_shape) return Tensor(grad);

    // 如果 target 是标量 [1] 而 grad 是多维的，直接求和
    if (target_shape.size() == 1 && target_shape[0] == 1) {
        float s = util::sum_all(grad);
        Tensor result = native::empty({1});
        result.data_ptr()[0] = s;
        return result;
    }

    // 左填充 target_shape 到与 grad 相同维度
    int ndim = grad_shape.size();
    int target_ndim = target_shape.size();
    std::vector<int> padded(ndim, 1);
    for (int i = 0; i < target_ndim; i++) {
        padded[ndim - target_ndim + i] = target_shape[i];
    }

    // 沿 target==1 但 grad>1 的维度求和
    Tensor result = grad.is_contiguous() ? Tensor(grad) : native::contiguous(grad);
    for (int i = 0; i < ndim; i++) {
        if (padded[i] == 1 && grad_shape[i] > 1) {
            result = util::sum_dim(result, i, true);
        }
    }
    // reshape 到 target_shape
    return native::reshape(result, std::vector<int>(target_shape));
}

} // namespace autograd_util

// MulScalarBackward: y = x * scalar
// grad_x = grad_output * scalar
class MulScalarBackward : public AutogradFunction {
public:
    float scalar;

    explicit MulScalarBackward(float s) : scalar(s) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {util::scalar_mul(grad_outputs[0], scalar)};
    }
};

// SumBackward: s = sum(x)
// grad_x = ones_like(x) * grad_output (scalar)
class SumBackward : public AutogradFunction {
public:
    std::vector<int> input_shape;

    explicit SumBackward(std::vector<int> shape)
        : input_shape(std::move(shape)) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        // grad 是标量张量（0-dim 或 1 个元素）
        float grad_val = grad.unsafeGetTensorImpl()->read_logical(0);
        Tensor result = native::fill(input_shape, grad_val);
        return {result};
    }
};

// AddBackward: z = broadcast_add(a, b)
// grad_a = reduce_grad(grad, a_shape)
// grad_b = reduce_grad(grad, b_shape)
class AddBackward : public AutogradFunction {
public:
    std::vector<int> a_shape;
    std::vector<int> b_shape;

    AddBackward(std::vector<int> as, std::vector<int> bs)
        : a_shape(std::move(as)), b_shape(std::move(bs)) {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        return {autograd_util::reduce_grad(grad, a_shape),
                autograd_util::reduce_grad(grad, b_shape)};
    }
};

// SubBackward: z = a - b
class SubBackward : public AutogradFunction {
public:
    std::vector<int> a_shape;
    std::vector<int> b_shape;

    SubBackward(std::vector<int> as, std::vector<int> bs)
        : a_shape(std::move(as)), b_shape(std::move(bs)) {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        Tensor grad_b = autograd_util::reduce_grad(grad, b_shape);
        // negate
        Tensor neg_b = util::scalar_mul(grad_b, -1.0f);
        return {autograd_util::reduce_grad(grad, a_shape), neg_b};
    }
};

// PassthroughBackward: gradient passes through unchanged (used for scalar add/sub)
class PassthroughBackward : public AutogradFunction {
public:
    PassthroughBackward() { num_inputs = 1; }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {Tensor(grad_outputs[0])};
    }
};

// NegBackward: y = -x
class NegBackward : public AutogradFunction {
public:
    NegBackward() { num_inputs = 1; }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {util::scalar_mul(grad_outputs[0], -1.0f)};
    }
};

// DivScalarBackward: y = x / scalar
class DivScalarBackward : public AutogradFunction {
public:
    float inv;

    explicit DivScalarBackward(float scalar) : inv(1.0f / scalar) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {util::scalar_mul(grad_outputs[0], inv)};
    }
};

// DivBackward: z = a / b
class DivBackward : public AutogradFunction {
public:
    Tensor saved_a;
    Tensor saved_b;
    std::vector<int> a_shape;
    std::vector<int> b_shape;

    DivBackward(Tensor a, Tensor b, std::vector<int> as, std::vector<int> bs)
        : saved_a(std::move(a)), saved_b(std::move(b)),
          a_shape(std::move(as)), b_shape(std::move(bs)) {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        // grad_a = grad / b
        Tensor ga = util::broadcast_div(grad, saved_b);
        // grad_b = -grad * a / b^2
        Tensor gb = util::broadcast_mul(grad, saved_a);
        Tensor b2 = util::broadcast_mul(saved_b, saved_b);
        gb = util::broadcast_div(gb, b2);
        gb = util::scalar_mul(gb, -1.0f);
        return {autograd_util::reduce_grad(ga, a_shape),
                autograd_util::reduce_grad(gb, b_shape)};
    }
};

// BroadcastMulBackward: z = a * b (with broadcast)
class BroadcastMulBackward : public AutogradFunction {
public:
    Tensor saved_a;
    Tensor saved_b;
    std::vector<int> a_shape;
    std::vector<int> b_shape;

    BroadcastMulBackward(Tensor a, Tensor b, std::vector<int> as, std::vector<int> bs)
        : saved_a(std::move(a)), saved_b(std::move(b)),
          a_shape(std::move(as)), b_shape(std::move(bs)) {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        Tensor ga = util::broadcast_mul(grad, saved_b);
        Tensor gb = util::broadcast_mul(grad, saved_a);
        return {autograd_util::reduce_grad(ga, a_shape),
                autograd_util::reduce_grad(gb, b_shape)};
    }
};

// MmBackward: z = a @ b (2D matmul)
class MmBackward : public AutogradFunction {
public:
    Tensor saved_a;
    Tensor saved_b;

    MmBackward(Tensor a, Tensor b)
        : saved_a(std::move(a)), saved_b(std::move(b)) {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        // grad_a = grad @ b^T
        Tensor bt = native::transpose(saved_b, 0, 1);
        Tensor ga = util::batched_matmul(grad, bt);
        // grad_b = a^T @ grad
        Tensor at = native::transpose(saved_a, 0, 1);
        Tensor gb = util::batched_matmul(at, grad);
        return {ga, gb};
    }
};

// BatchedMatmulBackward: z = batched_matmul(a, b)
class BatchedMatmulBackward : public AutogradFunction {
public:
    Tensor saved_a;
    Tensor saved_b;

    BatchedMatmulBackward(Tensor a, Tensor b)
        : saved_a(std::move(a)), saved_b(std::move(b)) {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        // grad_a = grad @ b^T
        Tensor bt = util::transpose_last2(saved_b);
        bt = bt.is_contiguous() ? bt : native::contiguous(bt);
        Tensor ga = util::batched_matmul(grad, bt);
        // grad_b = a^T @ grad
        Tensor at = util::transpose_last2(saved_a);
        at = at.is_contiguous() ? at : native::contiguous(at);
        Tensor gb = util::batched_matmul(at, grad);
        return {ga, gb};
    }
};

// TransposeBackward: y = transpose(x, d0, d1)
class TransposeBackward : public AutogradFunction {
public:
    int d0, d1;

    TransposeBackward(int d0_, int d1_) : d0(d0_), d1(d1_) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {native::transpose(grad_outputs[0], d0, d1)};
    }
};

// ViewBackward: y = reshape(x, new_shape)
class ViewBackward : public AutogradFunction {
public:
    std::vector<int> orig_shape;

    explicit ViewBackward(std::vector<int> shape)
        : orig_shape(std::move(shape)) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {native::reshape(grad_outputs[0], orig_shape)};
    }
};

// ExpandBackward: y = expand(x, new_sizes)
class ExpandBackward : public AutogradFunction {
public:
    std::vector<int> orig_shape;

    explicit ExpandBackward(std::vector<int> shape)
        : orig_shape(std::move(shape)) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {autograd_util::reduce_grad(grad_outputs[0], orig_shape)};
    }
};

// ReluBackward: y = relu(x)
class ReluBackward : public AutogradFunction {
public:
    Tensor saved_input;

    explicit ReluBackward(Tensor input) : saved_input(std::move(input)) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        Tensor result = native::empty(grad.sizes());
        float* pr = result.data_ptr();
        int n = grad.numel();
        auto* gi = grad.unsafeGetTensorImpl();
        auto* si = saved_input.unsafeGetTensorImpl();
        if (gi->is_contiguous() && si->is_contiguous()) {
            const float* pg = gi->data_ptr();
            const float* ps = si->data_ptr();
            for (int i = 0; i < n; i++) pr[i] = ps[i] > 0 ? pg[i] : 0.0f;
        } else {
            for (int i = 0; i < n; i++)
                pr[i] = si->read_logical(i) > 0 ? gi->read_logical(i) : 0.0f;
        }
        return {result};
    }
};

// TanhBackward: y = tanh(x), dy/dx = 1 - y^2
class TanhBackward : public AutogradFunction {
public:
    Tensor saved_output;

    explicit TanhBackward(Tensor output) : saved_output(std::move(output)) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        Tensor result = native::empty(grad.sizes());
        float* pr = result.data_ptr();
        int n = grad.numel();
        auto* gi = grad.unsafeGetTensorImpl();
        auto* oi = saved_output.unsafeGetTensorImpl();
        if (gi->is_contiguous() && oi->is_contiguous()) {
            const float* pg = gi->data_ptr();
            const float* po = oi->data_ptr();
            for (int i = 0; i < n; i++) pr[i] = pg[i] * (1.0f - po[i] * po[i]);
        } else {
            for (int i = 0; i < n; i++) {
                float o = oi->read_logical(i);
                pr[i] = gi->read_logical(i) * (1.0f - o * o);
            }
        }
        return {result};
    }
};

// SumDimBackward: y = sum(x, dim, keepdim)
class SumDimBackward : public AutogradFunction {
public:
    std::vector<int> orig_shape;
    int dim;
    bool keepdim;

    SumDimBackward(std::vector<int> shape, int d, bool kd)
        : orig_shape(std::move(shape)), dim(d), keepdim(kd) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        Tensor grad(grad_outputs[0]);
        // 如果 keepdim=false，先恢复被消去的维度
        if (!keepdim) {
            auto s = std::vector<int>(grad.sizes());
            s.insert(s.begin() + dim, 1);
            grad = native::reshape(grad, s);
        }
        // expand 到原始形状
        return {native::expand(grad, orig_shape)};
    }
};

// SliceBackward: y = x[dim, start:end]
class SliceBackward : public AutogradFunction {
public:
    std::vector<int> orig_shape;
    int dim, start;

    SliceBackward(std::vector<int> shape, int d, int s)
        : orig_shape(std::move(shape)), dim(d), start(s) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        // 创建全零张量，然后在对应位置填入梯度
        Tensor result = native::empty(orig_shape);
        float* pr = result.data_ptr();
        int slice_size = grad.sizes()[dim];

        // 计算外层/内层
        int ndim = orig_shape.size();
        int n_outer = 1, n_inner = 1;
        for (int i = 0; i < dim; i++) n_outer *= orig_shape[i];
        for (int i = dim + 1; i < ndim; i++) n_inner *= orig_shape[i];
        int full_dim = orig_shape[dim];

        Tensor cg = grad.is_contiguous() ? Tensor(grad) : native::contiguous(grad);
        const float* pg = cg.data_ptr();

        for (int outer = 0; outer < n_outer; outer++) {
            for (int d = 0; d < slice_size; d++) {
                for (int inner = 0; inner < n_inner; inner++) {
                    int src_idx = outer * slice_size * n_inner + d * n_inner + inner;
                    int dst_idx = outer * full_dim * n_inner + (start + d) * n_inner + inner;
                    pr[dst_idx] = pg[src_idx];
                }
            }
        }
        return {result};
    }
};

