#!/usr/bin/env python3
"""
模拟 PyTorch 的 torchgen 代码生成器
对应 PyTorch: torchgen/ 目录

读取 native_functions.yaml，生成:
  - generated/tensor_methods.h   Tensor 类的方法声明
  - generated/dispatch.h         Tensor 方法实现 (调用 native 函数)
  - generated/bindings.inl       pybind11 绑定代码
"""

import os
import re
import yaml

YAML_FILE = os.path.join(os.path.dirname(__file__), "native_functions.yaml")
GEN_DIR = os.path.join(os.path.dirname(__file__), "generated")

HEADER = """\
// ============================================================
// 自动生成的文件 — 请勿手动修改!
// 由 codegen.py 从 native_functions.yaml 生成
// ============================================================
"""


def parse_func(func_str):
    """解析函数签名，例如 'Tensor add(const Tensor& other) const'"""
    m = re.match(r"(\S+)\s+(\w+)\(([^)]*)\)\s*(const)?", func_str)
    if not m:
        raise ValueError(f"Cannot parse: {func_str}")
    return {
        "return_type": m.group(1),
        "name": m.group(2),
        "params": m.group(3),
        "const": m.group(4) or "",
    }


def generate():
    with open(YAML_FILE) as f:
        entries = yaml.safe_load(f)

    os.makedirs(GEN_DIR, exist_ok=True)

    # ---- generated/tensor_methods.h ----
    # Tensor 类内的方法声明
    lines = [HEADER, "#pragma once\n"]
    lines.append("// Tensor 算子方法声明 (插入 Tensor 类体内)\n")
    for entry in entries:
        parsed = parse_func(entry["func"])
        const_q = " const" if parsed["const"] else ""
        params = parsed["params"]
        lines.append(
            f"    {parsed['return_type']} {parsed['name']}({params}){const_q};\n"
        )
    with open(os.path.join(GEN_DIR, "tensor_methods.h"), "w") as f:
        f.writelines(lines)

    # ---- generated/dispatch.h ----
    # Tensor 方法实现，调用 native 函数
    lines = [HEADER, "#pragma once\n"]
    lines.append("// Tensor 方法实现 (调用 native 命名空间的 kernel)\n")
    for entry in entries:
        parsed = parse_func(entry["func"])
        const_q = " const" if parsed["const"] else ""
        params = parsed["params"]
        native_call = entry["native"]
        lines.append(
            f"inline {parsed['return_type']} Tensor::{parsed['name']}"
            f"({params}){const_q} {{ return {native_call}; }}\n"
        )
    with open(os.path.join(GEN_DIR, "dispatch.h"), "w") as f:
        f.writelines(lines)

    # ---- generated/tensor_bindings.inl ----
    # Tensor 类内的 pybind11 绑定（算子方法 + 运算符重载）
    lines = [HEADER]
    lines.append("// Tensor 类的算子方法绑定\n")
    for entry in entries:
        lines.append(f"        {entry['binding']}\n")
    lines.append("\n// Tensor 运算符重载绑定\n")
    for entry in entries:
        if "operator" in entry:
            lines.append(f"        {entry['operator']}\n")
    with open(os.path.join(GEN_DIR, "tensor_bindings.inl"), "w") as f:
        f.writelines(lines)

    # ---- generated/module_bindings.inl ----
    # 模块级算子函数绑定
    lines = [HEADER]
    lines.append("// 模块级算子函数绑定\n")
    for entry in entries:
        if "module_binding" in entry:
            lines.append(f"    {entry['module_binding']};\n")
    with open(os.path.join(GEN_DIR, "module_bindings.inl"), "w") as f:
        f.writelines(lines)

    print(f"[codegen] Generated {len(entries)} operators:")
    for entry in entries:
        parsed = parse_func(entry["func"])
        print(f"  - {parsed['name']}")
    print(f"[codegen] Output: {GEN_DIR}/")


if __name__ == "__main__":
    generate()
