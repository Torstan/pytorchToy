# pytorchToy

模拟 PyTorch 的核心 C++ 架构，包括：

- **`TensorImpl → TensorBase → Tensor` 三层类结构**
- **Storage / Stride / View 机制**（零拷贝的 transpose、slice、reshape、expand）
- **代码生成（codegen）**与**分离编译**减少增量编译时间
- **侵入式引用计数**（模板 `IntrusivePtr<T>`）
- **Autograd 自动微分**（C++ 反向传播引擎 + Python Variable/Function API）
- **版本计数 & In-place 检测**（save_for_backward 版本校验、mark_dirty 支持）

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
├── tensor_impl.h            # Storage + TensorImpl（含版本计数器）
├── tensor_base.h            # IntrusivePtr<T> 模板 + TensorImplPtr + TensorBase
├── tensor_base.cpp          # TensorBase 独立编译单元（repr 实现）
├── tensor.h                 # Tensor 类（包含 generated/tensor_methods.h）
├── ops.h                    # native 命名空间：算子 kernel + view 操作 + stride 辅助
├── bindings.cpp             # pybind11 Python 绑定层（Tensor + Autograd）
├── native_functions.yaml    # 算子定义（模拟 PyTorch 的 native_functions.yaml）
├── codegen.py               # 代码生成器（模拟 PyTorch 的 torchgen）
├── generated/               # 自动生成的文件（勿手动修改）
│   ├── tensor_methods.h     #   Tensor 方法声明
│   ├── dispatch.h           #   Tensor 方法实现（调用 native kernel）
│   ├── tensor_bindings.inl  #   pybind11 Tensor 算子绑定
│   └── module_bindings.inl  #   pybind11 模块级算子绑定
├── autograd/                # C++ Autograd 引擎
│   ├── function.h           #   AutogradFunction 基类 + InputInfo
│   ├── variable.h           #   VariableImpl（梯度存储、计算图链接）
│   ├── grad_ops.h           #   内置 backward 实现（Mul/Add/Sum/MulScalar）
│   ├── engine.h             #   反向传播引擎（拓扑排序 + 梯度分发）
│   └── py_function.h        #   PyFunction 桥接（C++ → Python backward 回调）
├── torch/                   # Python 包
│   ├── __init__.py          #   导出 Tensor, FloatTensor, zeros, ones, randn 等
│   ├── tensor.py            #   Python Tensor 包装类（含版本追踪）
│   └── autograd/
│       ├── __init__.py      #   导出 Variable, Function
│       ├── variable.py      #   Variable 类（backward, hooks, 运算符重载）
│       └── function.py      #   Function 基类（save_for_backward 版本检查, mark_dirty）
├── test/
│   └── test_autograd.py     # Autograd 测试（45 checks）
├── Makefile                 # 构建脚本（分离编译 + codegen）
├── demo.py                  # Python 演示 & 测试脚本
└── requirements.txt         # Python 依赖
```

---

## 3. Autograd 自动微分

### 架构概览

```
Python 层                              C++ 层
─────────                              ──────
Variable  ──── 持有 ────→  VariableImpl（data, grad, creator）
    │                           │
Function.__call__()             │
    │                           │
    ├─ forward() → 构建计算图    │
    │    save_for_backward()    │
    │    mark_dirty()           │
    │                           │
    └─ backward() ←── Engine ←──┘
         │              │
    PyFunction 桥接    拓扑排序 + 梯度分发
    (C++ ↔ Python)     (Kahn 算法)
```

### 计算图构建（前向）

每次 Variable 运算（`*`, `+`, `sum()`, 自定义 Function）会：
1. 计算前向结果
2. 创建对应的 `AutogradFunction` 节点（如 `MulBackward`, `PyFunction`）
3. 将输入边（`InputInfo`）连接到上游函数节点或叶子变量

### 反向传播（Engine）

`Engine::backward()` 从根节点出发：
1. **拓扑排序**（Kahn 算法）确定执行顺序
2. 按序调用每个节点的 `apply(grad_outputs)`
3. 将返回的 `grad_inputs` 分发给上游
4. 叶子节点的梯度累加到 `VariableImpl.grad`，并执行 hooks

### 内置 Backward 函数

| 节点 | Forward | Backward |
|------|---------|----------|
| `MulBackward` | `z = a * b` | `grad_a = grad * b, grad_b = grad * a` |
| `MulScalarBackward` | `y = x * s` | `grad_x = grad * s` |
| `AddBackward` | `z = a + b` | `grad_a = grad, grad_b = grad` |
| `SumBackward` | `s = sum(x)` | `grad_x = expand(grad, x.shape)` |

### 自定义 Function

```python
class Square(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x * x

    def backward(self, grad_output):
        x, = self.saved_tensors
        return 2 * x * grad_output

y = Square()(x)  # 前向 + 构建计算图
y.backward()      # 反向传播
```

### 版本计数 & In-place 检测

每个 `TensorImpl` 持有 `version_counter_`，在 in-place 操作时递增：

- `save_for_backward()` 记录张量及其当前版本号
- `saved_tensors` 访问时检查版本是否一致，不一致则抛出 `RuntimeError`
- `mark_dirty()` 声明 forward 中的 in-place 修改，使版本检查正确跳过

```python
class InplaceOp(Function):
    def forward(self, x):
        self.save_for_backward(x)
        self.mark_dirty(x)      # 声明 x 将被 in-place 修改
        x[0] = x[0] * 2         # in-place 修改
        return x
```

---

## 4. 算子与 View 操作一览

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

## 5. 编译依赖关系

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
  ← generated/*             │              ← generated/*
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

## 6. 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 首次构建
make all

# 运行演示 & 测试
make run

# 运行 autograd 测试（45 checks）
PYTHONPATH=. python3 test/test_autograd.py

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

## 7. 与 PyTorch 的对应关系

| 本项目 | PyTorch |
|--------|---------|
| `IntrusivePtr<T>` | `c10::intrusive_ptr<T>` |
| `TensorImplPtr` | `c10::intrusive_ptr<TensorImpl>` |
| `TensorImpl` | `c10/core/TensorImpl.h` |
| `TensorImpl::version_counter_` | `c10::VariableVersion` |
| `Storage` | `c10::StorageImpl` |
| `TensorBase` | `aten/src/ATen/core/TensorBase.h` |
| `Tensor` | `aten/src/ATen/templates/TensorBody.h` |
| `ops.h` (namespace native) | `aten/src/ATen/native/*.cpp` |
| `bindings.cpp` | `torch/csrc/` (pybind11 绑定) |
| `native_functions.yaml` | `aten/src/ATen/native/native_functions.yaml` |
| `codegen.py` | `torchgen/` |
| `AutogradFunction` | `torch::autograd::Node` |
| `Engine` | `torch::autograd::Engine` |
| `VariableImpl` | `torch::autograd::AutogradMeta` |
| `PyFunction` | `torch::autograd::PyNode` |
| `Variable` (Python) | `torch.autograd.Variable` (0.1.x) / `torch.Tensor` (modern) |
| `Function` (Python) | `torch.autograd.Function` |
| `save_for_backward` 版本检查 | `SavedVariable` 版本校验 |
| `mark_dirty` | `torch.autograd.Function.mark_dirty` |
