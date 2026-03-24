# pytorchToy

模拟 PyTorch 的核心 C++ 架构，包括：

- **`TensorImpl → TensorBase → Tensor` 三层类结构**
- **Storage / Stride / View 机制**（零拷贝的 transpose、slice、reshape、expand）
- **代码生成（codegen）**与**分离编译**减少增量编译时间
- **侵入式引用计数**（ 模板 `IntrusivePtr<T>`）

### 核心流程

```
native_functions.yaml  →  codegen.py  →  generated/*.h  →  编译链接
    (算子定义)            (代码生成)      (生成的C++代码)
```

---

## 1. 核心设计

### 三层类结构

对应 PyTorch 源码中的三层架构：

| 层级 | 本项目 | PyTorch 对应 | 职责 |
|------|--------|-------------|------|
| 底层 | `TensorImpl` | `c10/core/TensorImpl.h` | 持有数据（Storage）和元信息|
| 中层 | `TensorBase` | `aten/src/ATen/core/TensorBase.h` | 侵入式智能指针句柄，只有元信息方法 |
| 顶层 | `Tensor` | `aten/src/ATen/templates/TensorBody.h` | 继承 TensorBase，增加算子方法 |

### 为什么分离 TensorBase 和 Tensor？

> PyTorch 中大量基础设施代码只需要元信息（TensorBase），不需要算子。
> 把算子方法隔离到 Tensor 层，**修改算子时只重编译 Tensor 相关代码**，不影响依赖 TensorBase 的代码，大幅减少增量编译时间。

### Storage / Stride / View 机制

参考 [PyTorch internals (Edward Z. Yang)](https://blog.ezyang.com/2019/05/pytorch-internals/)：

> **Tensor = Storage（共享数据缓冲区）+ 元信息（sizes, strides, storage_offset）**

- **Storage**: 一块连续的 float 缓冲区，多个 TensorImpl 可以共享同一个 Storage
- **Strides**: 描述逻辑索引到 storage 偏移的映射，元素 `[i,j,k]` 的地址 = `storage_offset + i*stride[0] + j*stride[1] + k*stride[2]`
- **View 操作**: `transpose`、`slice`、`reshape`、`expand` 只修改元信息（sizes/strides/offset），不拷贝数据
- **Contiguous**: 当 strides 不等于行主序标准 strides 时，tensor 是 non-contiguous 的，可调用 `contiguous()` 拷贝为连续存储

```
Tensor a (2x3)           Tensor at = a.transpose(0,1) (3x2)
  sizes=[2,3]              sizes=[3,2]
  strides=[3,1]            strides=[1,3]      ← 交换 strides，零拷贝
  offset=0                 offset=0
       │                        │
       └───── 共享同一块 Storage [0,1,2,3,4,5] ─────┘
```

### 侵入式引用计数

```
IntrusivePtr<T>             ← 通用模板智能指针
TensorImplPtr               ← IntrusivePtr<TensorImpl> 的类型别名
```

与 `shared_ptr` 的区别：引用计数存储在对象自身中，少一次控制块的内存分配。

---

## 2. 文件结构

```
pytorchToy/
├── tensor_impl.h            # Storage + TensorImpl
├── tensor_base.h            # IntrusivePtr<T> 模板 + TensorImplPtr + TensorBase
├── tensor_base.cpp          # TensorBase 独立编译单元（repr 实现）
├── tensor.h                 # Tensor 类（包含 generated/tensor_methods.h）
├── ops.h                    # native 命名空间：算子 kernel + view 操作 + stride 辅助
├── bindings.cpp             # pybind11 Python 绑定层
├── native_functions.yaml    # 算子定义（模拟 PyTorch 的 native_functions.yaml）
├── codegen.py               # 代码生成器（模拟 PyTorch 的 torchgen）
├── generated/               # 自动生成的文件（勿手动修改）
│   ├── tensor_methods.h     #   Tensor 方法声明
│   ├── dispatch.h           #   Tensor 方法实现（调用 native kernel）
│   ├── tensor_bindings.inl  #   pybind11 Tensor 算子绑定
│   └── module_bindings.inl  #   pybind11 模块级算子绑定
├── Makefile                 # 构建脚本（分离编译 + codegen）
├── demo.py                  # Python 演示 & 测试脚本
└── requirements.txt         # Python 依赖
```

---

## 3. 算子与 View 操作一览

### 计算算子

| 算子 | 说明 | Stride 感知 |
|------|------|------------|
| `add(a, b)` | 逐元素加法 | ✓ |
| `mul(a, b)` | 逐元素乘法 | ✓ |
| `matmul(a, b)` | 2D 矩阵乘法 | ✓ |
| `relu(a)` | ReLU 激活 | ✓ |
| `sum(a)` | 全元素求和 | ✓ |

### View 操作（零拷贝，共享 Storage）

| 操作 | 说明 |
|------|------|
| `transpose(dim0, dim1)` | 交换两个维度的 size 和 stride |
| `slice(dim, start, end)` | 取指定维度的 [start, end) 子区间 |
| `reshape(new_shape)` | 改变形状（contiguous 时零拷贝，支持 -1 推断） |
| `expand(new_sizes)` | 将 size=1 的维度广播（stride 设为 0） |
| `contiguous()` | 若已连续直接返回，否则拷贝为连续存储 |

### 索引

| 表达式 | 结果 |
|--------|------|
| `t[i]` | 降一维的 view tensor |
| `t[i,j]` | 降两维的 view tensor |
| `t[i,j,k]`（完全索引） | 0-dim 标量 tensor，用 `.item()` 取值 |

### 工厂函数

| 函数 | 说明 |
|------|------|
| `tensor(nested_list)` | 从嵌套 list 创建（推断 shape） |
| `Tensor.from_data(nested_list)` | 同上，静态方法形式 |
| `ones(shape)` | 全 1 tensor |
| `fill(shape, value)` | 指定值填充 |
| `empty(shape)` | 全 0 tensor |

---

## 4. 编译依赖关系

```
                    native_functions.yaml
                            │
                        codegen.py
                            │
                      generated/*.h
                            │
tensor_base.o               │            bindings.o
  ← tensor_base.cpp         │              ← bindings.cpp
  ← tensor_base.h           │              ← ops.h, tensor.h
  ← tensor_impl.h           │              ← tensor_base.h, tensor_impl.h
  (不依赖 generated/*)      │              ← generated/*
          │                                      │
          └──────────── _C.so ───────────────────┘
```

**修改 `native_functions.yaml` 后：**

| 编译目标 | 是否重编译 | 原因 |
|---------|-----------|------|
| `codegen.py` | 重新执行 | YAML 变更触发代码生成 |
| `bindings.o` | 重新编译 | 依赖 `generated/*` |
| `tensor_base.o` | **不重新编译** | 不依赖算子定义 |

---

## 5. 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 首次构建
make all

# 运行演示 & 测试
make run

# 验证增量编译（观察 tensor_base.o 不会被重编译）
make demo_codegen

# 清理
make clean
```

### 新增算子的步骤

1. 在 `native_functions.yaml` 中添加算子定义
2. 在 `ops.h` 的 `namespace native` 中实现 kernel
3. 执行 `make all`（codegen 自动生成声明和绑定）

---

## 6. 与 PyTorch 的对应关系

| 本项目 | PyTorch |
|--------|---------|
| `IntrusivePtr<T>` | `c10::intrusive_ptr<T>` |
| `TensorImplPtr` | `c10::intrusive_ptr<TensorImpl>` |
| `TensorImpl` | `c10/core/TensorImpl.h` |
| `Storage` | `c10::StorageImpl` |
| `TensorBase` | `aten/src/ATen/core/TensorBase.h` |
| `Tensor` | `aten/src/ATen/templates/TensorBody.h` |
| `ops.h` (namespace native) | `aten/src/ATen/native/*.cpp` |
| `bindings.cpp` | `torch/csrc/` (pybind11 绑定) |
| `native_functions.yaml` | `aten/src/ATen/native/native_functions.yaml` |
| `codegen.py` | `torchgen/` |
