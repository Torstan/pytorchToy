#pragma once

#include "function.h"
#include "../ops.h"

// ============================================================
// 内置算子的 backward 实现
// 这些是 Variable 运算符重载（__mul__, sum() 等）
// 在前向过程中自动创建的计算图节点
// ============================================================

// MulBackward: z = a * b
// grad_a = grad_output * b
// grad_b = grad_output * a
class MulBackward : public AutogradFunction {
public:
    Tensor saved_a;
    Tensor saved_b;

    MulBackward(Tensor a, Tensor b)
        : saved_a(std::move(a)), saved_b(std::move(b)) {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        Tensor grad_a = native::mul(grad, saved_b);
        Tensor grad_b = native::mul(grad, saved_a);
        return {grad_a, grad_b};
    }
};

// MulScalarBackward: y = x * scalar
// grad_x = grad_output * scalar
class MulScalarBackward : public AutogradFunction {
public:
    float scalar;

    explicit MulScalarBackward(float s) : scalar(s) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        // grad * scalar
        Tensor result = native::empty(std::vector<int>(grad.sizes()));
        float* pr = result.data_ptr();
        auto* impl = grad.unsafeGetTensorImpl();
        for (int i = 0; i < grad.numel(); i++) {
            pr[i] = impl->read_logical(i) * scalar;
        }
        return {result};
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

// AddBackward: z = a + b
// grad_a = grad_output
// grad_b = grad_output
class AddBackward : public AutogradFunction {
public:
    AddBackward() {
        num_inputs = 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad = grad_outputs[0];
        // 两个输入的梯度都是 grad_output 本身
        // 需要拷贝以避免共享
        Tensor grad_a = native::empty(std::vector<int>(grad.sizes()));
        Tensor grad_b = native::empty(std::vector<int>(grad.sizes()));
        float* pa = grad_a.data_ptr();
        float* pb = grad_b.data_ptr();
        auto* impl = grad.unsafeGetTensorImpl();
        for (int i = 0; i < grad.numel(); i++) {
            float v = impl->read_logical(i);
            pa[i] = v;
            pb[i] = v;
        }
        return {grad_a, grad_b};
    }
};
