#pragma once

#include "../tensor.h"
#include <memory>
#include <vector>
#include <utility>

// 前向声明
struct VariableImpl;

// ============================================================
// AutogradFunction — 计算图节点基类
// 对应 PyTorch 的 torch::autograd::Function
//
// 每个 Function 节点代表一次前向运算。
// inputs 记录每个输入的来源（上游函数或叶子变量）。
// apply() 执行反向传播，计算 grad_inputs。
// ============================================================

// 每个输入的信息
struct InputInfo {
    std::shared_ptr<AutogradFunction> fn;  // 上游函数节点（叶子变量为 nullptr）
    int output_index = 0;                   // 在上游函数输出中的索引
    VariableImpl* variable = nullptr;       // 叶子变量（非叶子为 nullptr）
};

class AutogradFunction : public std::enable_shared_from_this<AutogradFunction> {
public:
    // 输入列表：每个输入要么来自上游函数（fn != null），
    // 要么是叶子变量（variable != null）
    std::vector<InputInfo> inputs;

    // 输入个数
    int num_inputs = 0;

    // 是否需要梯度
    bool requires_grad = true;

    // 反向传播接口 — 由子类实现
    // 输入: grad_outputs（对输出的梯度）
    // 输出: grad_inputs（对输入的梯度）
    virtual std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) = 0;

    virtual ~AutogradFunction() = default;
};
