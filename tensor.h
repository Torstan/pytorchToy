#pragma once
#include "tensor_base.h"

// ============================================================
// Tensor — 继承 TensorBase，增加算子方法
// 对应 PyTorch: aten/src/ATen/templates/TensorBody.h
//
// Tensor 在 TensorBase 基础上增加了由代码生成器自动生成的算子方法
// （add, mul, matmul 等）。这些方法依赖 native_functions.yaml。
// 当 native_functions.yaml 变更时，只需重编译 Tensor 相关代码，
// 而不需要重编译只依赖 TensorBase 的代码。
// ============================================================

class Tensor : public TensorBase {
public:
    Tensor() = default;
    explicit Tensor(TensorImplPtr impl) : TensorBase(std::move(impl)) {}
    Tensor(Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;

    // 从 TensorBase 构造（需要增加引用计数）
    explicit Tensor(const TensorBase& base) : TensorBase(base) {}
    /*implicit*/ Tensor(TensorBase&& base) : TensorBase(std::move(base)) {}

    // ---- 以下方法由 codegen.py 从 native_functions.yaml 自动生成 ----
    // 在 PyTorch 中这些是由 torchgen 从 native_functions.yaml 自动生成的
    // 声明在此，实现在 ops.h 中（避免循环依赖）
#include "generated/tensor_methods.h"
};
