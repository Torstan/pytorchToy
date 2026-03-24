// ============================================================
// 自动生成的文件 — 请勿手动修改!
// 由 codegen.py 从 native_functions.yaml 生成
// ============================================================
#pragma once
// Tensor 算子方法声明 (插入 Tensor 类体内)
    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor relu() const;
    float sum() const;
    Tensor contiguous() const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor slice(int dim, int start, int end) const;
    Tensor reshape(std::vector<int> shape) const;
    Tensor expand(std::vector<int> sizes) const;
