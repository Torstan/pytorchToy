#pragma once

#include "../tensor.h"
#include <memory>
#include <vector>
#include <functional>

// 前向声明
class AutogradFunction;

// ============================================================
// VariableImpl — 变量的 autograd 元数据
// 对应 PyTorch 0.1.x 中 Variable 的内部实现
//
// 持有底层 Tensor、梯度 Tensor、计算图链接（creator）
// 以及 hooks 列表。
// ============================================================

struct VariableImpl {
    Tensor data;                 // 底层张量数据
    Tensor grad;                 // 累加的梯度张量
    bool grad_defined = false;   // grad 是否已经初始化

    bool requires_grad = false;
    bool is_volatile = false;

    // 计算图链接
    std::shared_ptr<AutogradFunction> creator;  // 产生此变量的函数节点
    int output_index = 0;                       // 在 creator 输出中的索引

    // 梯度 hooks：backward 时对梯度的后处理
    std::vector<std::function<void(Tensor&)>> hooks;

    // 构造函数
    VariableImpl() = default;

    explicit VariableImpl(Tensor data_, bool requires_grad_ = false, bool volatile_ = false)
        : data(std::move(data_))
        , requires_grad(requires_grad_)
        , is_volatile(volatile_) {}

    // 累加梯度
    void accumulate_grad(const Tensor& grad_to_add) {
        if (!grad_defined) {
            // 首次设置：创建同形状零张量并加上
            grad = Tensor(TensorImplPtr(new TensorImpl(
                std::vector<int>(grad_to_add.sizes()), 0.0f)));
            grad_defined = true;
        }
        // 逐元素累加
        float* dst = grad.data_ptr();
        int n = grad.numel();
        // grad_to_add 可能不连续，使用 read_logical
        auto* src_impl = grad_to_add.unsafeGetTensorImpl();
        for (int i = 0; i < n; i++) {
            dst[i] += src_impl->read_logical(i);
        }
    }

    // 执行 hooks
    void run_hooks() {
        for (auto& hook : hooks) {
            hook(grad);
        }
    }
};
