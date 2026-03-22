1. 主要功能：

模拟 PyTorch 的 TensorImpl → TensorBase → Tensor 三层类架构，
演示代码生成（codegen）和分离编译如何减少增量编译时间。

核心流程：
  native_functions.yaml  →  codegen.py  →  generated/*.h  →  编译链接
  （算子定义）              （代码生成）     （生成的C++代码）

2. 核心设计

三层类结构（对应 PyTorch 源码）：

  TensorImpl (tensor_impl.h)    ← c10/core/TensorImpl.h
    - 持有真正的数据（Storage）和元信息（sizes, strides, device）
    - 引用计数对象，多个 Tensor 可共享同一个 TensorImpl
    - storage_offset_ 支持零拷贝视图（view/slice）

  TensorBase (tensor_base.h)    ← aten/src/ATen/core/TensorBase.h
    - 持有 IntrusivePtr<TensorImpl>（侵入式智能指针）
    - 只有元信息方法（dim, sizes, strides, device 等）
    - 不依赖 native_functions.yaml → 修改算子不需要重编译

  Tensor (tensor.h)             ← aten/src/ATen/templates/TensorBody.h
    - 继承 TensorBase，增加算子方法（add, mul, matmul 等）
    - 算子方法声明由 codegen.py 从 native_functions.yaml 自动生成

为什么分离 TensorBase 和 Tensor？
  PyTorch 中大量基础设施代码只需要元信息（TensorBase），不需要算子。
  把算子方法隔离到 Tensor 层，修改算子时只重编译 Tensor 相关代码，
  不影响依赖 TensorBase 的代码，大幅减少增量编译时间。

3. 文件结构

  tensor_impl.h          底层数据结构（Storage + 元信息 + 引用计数）
  tensor_base.h          TensorBase 类 + IntrusivePtr（不依赖算子）
  tensor_base.cpp        TensorBase 独立编译单元（repr 实现）
  tensor.h               Tensor 类（包含 generated/tensor_methods.h）
  ops.h                  native 命名空间的算子 kernel 实现
  bindings.cpp           pybind11 Python 绑定层
  native_functions.yaml  算子定义（模拟 PyTorch 的 native_functions.yaml）
  codegen.py             代码生成器（模拟 PyTorch 的 torchgen）
  generated/             自动生成的文件（勿手动修改）
    tensor_methods.h       Tensor 方法声明
    dispatch.h             Tensor 方法实现（调用 native kernel）
    tensor_bindings.inl    pybind11 Tensor 算子绑定
    module_bindings.inl    pybind11 模块级算子绑定
  Makefile               构建脚本（分离编译 + codegen）
  demo.py                Python 演示脚本

4. 编译依赖关系

  tensor_base.o  ← tensor_base.cpp, tensor_base.h, tensor_impl.h
                    不依赖: native_functions.yaml, ops.h, generated/*

  bindings.o     ← bindings.cpp, ops.h, tensor.h, tensor_base.h,
                    tensor_impl.h, generated/*
                    依赖: native_functions.yaml（通过 generated/*）

  _C.so          ← tensor_base.o + bindings.o（链接）

  修改 native_functions.yaml 后:
    codegen.py 重新生成  → bindings.o 重新编译  → 重新链接
    tensor_base.o 不受影响，不重新编译

5. 使用方法

  # 首次构建
  make all

  # 运行演示
  make run

  # 验证增量编译（观察 tensor_base.o 不会被重编译）
  make demo_codegen

  # 清理
  make clean

  # 新增算子的步骤:
  # 1. 在 native_functions.yaml 中添加算子定义
  # 2. 在 ops.h 的 namespace native 中实现 kernel
  # 3. make all（codegen 自动生成声明和绑定）

6. 与 PyTorch 的对应关系

  本项目                      PyTorch
  ─────────────────────────────────────────────────
  tensor_impl.h               c10/core/TensorImpl.h
  IntrusivePtr                c10::intrusive_ptr
  tensor_base.h               aten/src/ATen/core/TensorBase.h
  tensor.h                    aten/src/ATen/templates/TensorBody.h
  ops.h (namespace native)    aten/src/ATen/native/*.cpp
  bindings.cpp                torch/csrc/ (pybind11 绑定)
  native_functions.yaml       aten/src/ATen/native/native_functions.yaml
  codegen.py                  torchgen/
