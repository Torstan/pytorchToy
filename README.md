# pytorchToy

**pytorchToy** is an educational PyTorch implementation that faithfully reproduces
PyTorch's core C++ architecture from scratch. It covers the full stack from
low-level tensor storage and stride-based views, through code generation and
compilation firewalls, to a complete neural network module system with C++ kernels
— all in ~5,000 lines of C++ and Python.

- [Highlighted Features](#highlighted-features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Tour](#quick-tour)
  - [Tensor Operations](#tensor-operations)
  - [Training an RNN](#training-an-rnn)
  - [Training a Transformer](#training-a-transformer)
- [Core Design](#core-design)
  - [Three-Layer Tensor Hierarchy](#three-layer-tensor-hierarchy)
  - [Storage / Stride / View](#storage--stride--view)
  - [Intrusive Reference Counting](#intrusive-reference-counting)
- [nn Module System](#nn-module-system)
- [Autograd](#autograd)
- [Operator Reference](#operator-reference)
- [Project Structure](#project-structure)
- [Extending pytorchToy](#extending-pytorchtoy)
- [Performance Notes](#performance-notes)
- [PyTorch Source Mapping](#pytorch-source-mapping)

## Highlighted Features

* **Three-layer tensor architecture**

  Faithfully reproduces PyTorch's `TensorImpl → TensorBase → Tensor` class
  hierarchy with a compilation firewall — modifying operators only recompiles
  `bindings.o`, not `tensor_base.o`.

* **Stride-based view system**

  Zero-copy `transpose`, `slice`, `reshape`, `expand` via shared Storage
  and stride manipulation. The `expand` operation uses the stride=0 trick,
  identical to real PyTorch.

* **Code generation pipeline**

  `native_functions.yaml → codegen.py → generated/*.h`, mirroring
  PyTorch's `torchgen` workflow. Operators are declared in YAML and
  automatically wired into C++ dispatch and Python bindings.

* **Dual autograd systems**

  System A: C++ Variable-based engine with topological-sort backward pass
  (PyTorch 0.1–0.3 style). System B: Python graph-based tape autograd where
  each Tensor carries `_grad_fn` (PyTorch 1.0+ style). Both fully functional.

* **Complete nn module system**

  `Module`, `Parameter`, `Linear`, `RNN`, `Embedding`, `LayerNorm`,
  `MultiheadAttention`, full `Transformer` (Encoder + Decoder),
  `CrossEntropyLoss`, and `Adam` optimizer — all with C++ forward/backward kernels
  exposed via a dedicated `_nn_C.so` extension.

* **End-to-end training**

  RNN character-sequence prediction (loss converges from 0.003 → 0.0001) and
  Transformer forward + backward + optimizer step, both running successfully.

* **Easy to extend**

  Adding a new core operator requires only a YAML entry + a C++ kernel function.
  Adding a new nn layer requires a C++ forward/backward in `nn/ops.h`,
  a pybind11 binding, and a Python Module class.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Layer (torch/)                                          │
│                                                                 │
│  torch.Tensor ─── torch.nn.Module ─── torch.optim.Adam         │
│       │               │                                         │
│  autograd_engine.py   │  (Graph-based Tape Autograd)            │
│       │               │                                         │
├───────┼───────────────┼─────────────────────────────────────────┤
│  pybind11 Bindings                                              │
│                                                                 │
│  _C.so (Tensor + C++ Autograd)    _nn_C.so (nn kernels)        │
│       │                                │                        │
├───────┼────────────────────────────────┼────────────────────────┤
│  C++ Layer                             │                        │
│                                        │                        │
│  TensorImpl ← TensorBase ← Tensor     nn/ops.h (forward/bwd)  │
│       │                                util/math.h             │
│  Storage (shared_ptr)                  util/tensor_ops.h       │
│  IntrusivePtr<T> (ref counting)                                 │
│                                                                 │
│  native_functions.yaml → codegen.py → generated/*.h             │
└─────────────────────────────────────────────────────────────────┘
```

Two shared libraries are built:
- **`_C.so`** — Core tensor operations, view mechanics, and C++ autograd engine
- **`_nn_C.so`** — All neural network kernels (linear, rnn, embedding, layernorm, attention, cross-entropy, etc.)

## Installation

### Requirements

- Linux (tested on WSL2)
- g++ with C++17 support
- Python >= 3.8
- pybind11

### Build

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build both _C.so and _nn_C.so
make all

# Verify
python3 test/demo.py
```

### Build from source (step by step)

```bash
# 1. Code generation: native_functions.yaml → generated/*.h
make codegen

# 2. Compile and link
make all

# 3. Clean rebuild
make clean && make all
```

## Quick Tour

### Tensor Operations

```python
import _C

# Create tensors
a = _C.tensor([[1, 2, 3], [4, 5, 6]])   # 2x3
b = a.transpose(0, 1)                     # 3x2, zero-copy view

# Shared storage
assert a.data_ptr() == b.data_ptr()

# Stride-aware operations
c = a.slice(1, 0, 2)                      # 2x2, zero-copy
d = a.reshape([3, 2])                      # 3x2, zero-copy if contiguous
```

### Training an RNN

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Character-sequence prediction: "hello" → "elloh"
model = nn.RNN(input_size=vocab_size, hidden_size=16, num_layers=1)
linear = nn.Linear(16, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() + linear.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    output, hidden = model(input_seq)
    loss = criterion(output.view(-1, vocab_size), target)
    loss.backward()
    optimizer.step()
# → Loss: 0.003 → 0.0001, predicts "elloh" correctly
```

```bash
python3 test/rnn.py
```

### Training a Transformer

```python
import torch
import torch.nn as nn

model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=2)
src = torch.randn([10, 32, 512])   # [seq_len, batch, d_model]
tgt = torch.randn([20, 32, 512])

output = model(src, tgt)            # forward
loss = output.sum()
loss.backward()                     # backward through entire Transformer
```

```bash
python3 test/transformer.py
```

## Core Design

### Three-Layer Tensor Hierarchy

| Layer | This Project | PyTorch Equivalent | Role |
|-------|-------------|-------------------|------|
| Bottom | `TensorImpl` | `c10/core/TensorImpl.h` | Holds Storage + metadata (sizes, strides, offset) |
| Middle | `TensorBase` | `aten/src/ATen/core/TensorBase.h` | Intrusive-ptr handle, metadata-only methods |
| Top | `Tensor` | `aten/src/ATen/templates/TensorBody.h` | Inherits TensorBase, adds operator methods via codegen |

**Compilation firewall**: `tensor_base.o` does not depend on `ops.h` or `generated/*`.
Modifying operators triggers recompilation of `bindings.o` only.

### Storage / Stride / View

Ref: [PyTorch internals (Edward Z. Yang, 2019)](https://blog.ezyang.com/2019/05/pytorch-internals/)

> **Tensor = Storage (shared data buffer) + metadata (sizes, strides, storage_offset)**

```
Tensor a (2x3)           Tensor b = a.transpose(0,1) (3x2)
  sizes=[2,3]              sizes=[3,2]
  strides=[3,1]            strides=[1,3]      ← swap strides, zero-copy
  offset=0                 offset=0
       │                        │
       └───── shared Storage [0,1,2,3,4,5] ─────┘
```

- **transpose**: swaps sizes and strides, zero-copy
- **slice**: adjusts storage_offset, zero-copy
- **reshape**: recomputes strides if contiguous, zero-copy
- **expand**: sets stride=0 for broadcast dimensions, zero-copy

### Intrusive Reference Counting

```
IntrusivePtr<T>             ← Generic template smart pointer
TensorImplPtr               ← IntrusivePtr<TensorImpl> alias
```

Reference count lives inside the object itself (`IntrusivePtrTarget`),
avoiding the extra control-block allocation of `std::shared_ptr`.

## nn Module System

All nn layers call C++ kernels for compute-heavy operations and use
Python tape autograd (`autograd_engine.py`) for gradient tracking:

```python
# Inside torch/nn/linear.py:
output_c = _nn_C.linear_forward(input._c, self.weight._c, self.bias._c)
record([output], [input, self.weight, self.bias], backward_fn)
```

### Implemented Layers

| Module | C++ Kernel | Description |
|--------|-----------|-------------|
| `nn.Module` | — | Base class with parameter management, recursive traversal |
| `nn.Parameter` | — | Tensor subclass with `requires_grad=True` |
| `nn.Linear` | `linear_forward/backward` | Fully-connected layer |
| `nn.RNN` | `rnn_forward/backward` | Elman RNN, multi-layer, BPTT |
| `nn.Embedding` | `embedding_forward/backward` | Lookup table |
| `nn.LayerNorm` | `layer_norm_forward/backward` | Layer normalization |
| `nn.MultiheadAttention` | `multihead_attention_forward` | Multi-head attention |
| `nn.TransformerEncoder` | — | Stacked encoder layers |
| `nn.TransformerDecoder` | — | Stacked decoder layers |
| `nn.Transformer` | — | Full Encoder-Decoder Transformer |
| `nn.CrossEntropyLoss` | `cross_entropy_forward/backward` | Cross-entropy loss |
| `optim.Adam` | — | Adam optimizer with bias correction |

### C++ Utility Kernels

| Function | Location | Description |
|----------|----------|-------------|
| `batched_matmul` | `util/tensor_ops.h` | 2D/3D/4D matmul with broadcasting |
| `softmax / log_softmax` | `util/tensor_ops.h` | Numerically stable |
| `broadcast_add/mul/sub/div` | `util/math.h` | Element-wise with broadcasting |
| `elementwise_relu/tanh/exp/log` | `util/math.h` | Activation functions |
| `sum_dim / mean_dim / var_dim` | `util/math.h` | Reduction operations |
| `fill_randn / fill_uniform` | `util/math.h` | Random initialization |
| `cat / chunk / transpose_last2` | `util/math.h` | Tensor manipulation |

## Autograd

Two autograd systems coexist:

* **System A: C++ Variable-based** (PyTorch 0.1–0.3 style)

  `Variable` wraps a `Tensor` and links to a `VariableImpl` in C++.
  `Function` subclasses define forward/backward. The C++ `Engine` performs
  backward via Kahn's algorithm topological sort. Supports `save_for_backward`
  version checking and `mark_dirty` for in-place ops.

* **System B: Python graph-based tape** (PyTorch 1.0+ style)

  Used by the nn module system. Each `Tensor` carries `_grad_fn` → `GradFn` node.
  `backward()` does DFS topological sort, then reverse traversal accumulating
  gradients to leaf parameters.

<details>
<summary>Custom Function example (System A)</summary>

```python
from torch.autograd import Function

class Square(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x * x

    def backward(self, grad_output):
        x, = self.saved_tensors
        return 2 * x * grad_output

y = Square()(x)
y.backward()
```
</details>

## Operator Reference

### Compute Operators (`_C` module)

| Operator | Description | Stride-aware |
|----------|-------------|:------------:|
| `add(a, b)` | Element-wise addition | Yes |
| `mul(a, b)` | Element-wise multiplication | Yes |
| `matmul(a, b)` | 2D matrix multiplication | Yes |
| `relu(a)` | ReLU activation | Yes |
| `sum(a)` | Full reduction | Yes |

### View Operations (zero-copy, shared Storage)

| Operation | Description |
|-----------|-------------|
| `transpose(dim0, dim1)` | Swap two dimensions' size and stride |
| `slice(dim, start, end)` | Sub-interval `[start, end)` on a dimension |
| `reshape(new_shape)` | Change shape (zero-copy if contiguous, `-1` inference) |
| `expand(new_sizes)` | Broadcast size=1 dims (stride set to 0) |
| `contiguous()` | Copy to contiguous storage if needed |

### Factory Functions

| Function | Description |
|----------|-------------|
| `torch.tensor(data)` | Create from nested list (infer shape) |
| `torch.zeros(shape)` | All zeros |
| `torch.ones(shape)` | All ones |
| `torch.randn(shape)` | Standard normal random |
| `torch.randint(low, high, shape)` | Random integers |
| `torch.manual_seed(seed)` | Set random seed |

## Project Structure

```
pytorchToy/
├── tensor_impl.h              # Storage + TensorImpl (version counter)
├── intrusive_ptr.h            # IntrusivePtr<T> template
├── tensor_base.h/.cpp         # TensorBase (metadata-only handle)
├── tensor.h                   # Tensor (includes generated/tensor_methods.h)
├── ops.h                      # Native kernels + view operations
├── bindings.cpp               # pybind11 bindings (_C module)
├── native_functions.yaml      # Operator definitions
├── codegen.py                 # Code generator (→ generated/*.h)
├── Makefile                   # Build system
│
├── autograd/                  # C++ autograd engine
│   ├── function.h             #   AutogradFunction base + InputInfo
│   ├── variable.h             #   VariableImpl (grad, creator link)
│   ├── grad_ops.h             #   Built-in backward ops
│   ├── engine.h               #   Backward engine (topo-sort)
│   └── py_function.h          #   C++ ↔ Python bridge
│
├── nn/                        # C++ nn kernels
│   ├── ops.h                  #   All forward + backward implementations
│   └── nn_bindings.cpp        #   pybind11 bindings (_nn_C module)
│
├── util/                      # C++ utilities
│   ├── math.h                 #   Broadcasting, elementwise, reduction, random
│   └── tensor_ops.h           #   batched_matmul, softmax, log_softmax
│
├── torch/                     # Python package (import torch)
│   ├── tensor.py              #   Tensor wrapper (ops, indexing, autograd)
│   ├── autograd_engine.py     #   Graph-based tape autograd
│   ├── autograd/              #   Variable / Function API
│   ├── nn/                    #   Module, Linear, RNN, Embedding, Transformer, ...
│   └── optim/                 #   Adam optimizer
│
└── test/
    ├── demo.py                #   Tensor architecture demo
    ├── test_autograd.py       #   C++ autograd tests (45 checks)
    ├── rnn.py                 #   RNN training example
    ├── transformer.py         #   Transformer example
    └── bench_matmul.py        #   matmul performance benchmark
```

### Compilation Dependency Graph

```
native_functions.yaml → codegen.py → generated/*.h
                                          │
tensor_base.o ──────────────┐             │
  (tensor_base.cpp)         │         bindings.o
  (tensor_impl.h)           │           (ops.h, generated/*)
                            ▼             │
                         _C.so ◄──────────┘

nn/ops.h + util/*.h → nn_bindings.o → _nn_C.so
```

> Modifying operators recompiles `bindings.o` only — `tensor_base.o` is untouched.

## Extending pytorchToy

### Adding a core operator (`_C` module)

1. Add entry to `native_functions.yaml`
2. Implement kernel in `ops.h` (`namespace native`)
3. Run `make all` — codegen generates declarations and bindings automatically

### Adding an nn layer (`_nn_C` module)

1. Implement C++ forward + backward in `nn/ops.h`
2. Add pybind11 binding in `nn/nn_bindings.cpp`
3. Write Python Module class in `torch/nn/`
4. Run `make all`

## Performance Notes

`batched_matmul` was systematically benchmarked across 5 variants × 2 compiler
optimization levels. Full data in `test/bench_results.txt`.

| Optimization | -O0 | -O2 -march=native |
|-------------|-----|-------------------|
| i-k-j loop reorder | Slower (+6~153%) | Slower (+6~57%) |
| 3D/4D fast path | Slower | **Faster (-8~-64%)** |
| Tiling 32x32x32 | Slower | Slower (+19~104%) |
| `__restrict__`/ivdep | Neutral | Dragged by tiling |

**Final choice: V2 (i-k-j + fast paths, no tiling)** — up to 56-64% speedup on
large matrices under `-O2 -march=native`.

<details>
<summary>Key findings</summary>

- Under `-O2`, the compiler vectorizes the simple 3-loop i-k-j kernel very well;
  tiling's 6-loop structure actually hinders auto-vectorization
- Fast paths (skipping `flat_to_coords`/`broadcast_flat_idx`) provide the main
  benefit for 3D/4D batched operations
- All manual optimizations are **negative** under `-O0` — function call overhead,
  zeroing, and extra loop control are fully exposed without compiler optimization
</details>

## PyTorch Source Mapping

| pytorchToy | PyTorch |
|-----------|---------|
| `IntrusivePtr<T>` | `c10::intrusive_ptr<T>` |
| `TensorImpl` | `c10/core/TensorImpl.h` |
| `Storage` | `c10::StorageImpl` |
| `TensorBase` | `aten/src/ATen/core/TensorBase.h` |
| `Tensor` | `aten/src/ATen/templates/TensorBody.h` |
| `ops.h` | `aten/src/ATen/native/*.cpp` |
| `native_functions.yaml` | `aten/src/ATen/native/native_functions.yaml` |
| `codegen.py` | `torchgen/` |
| `nn/ops.h` | `aten/src/ATen/native/` |
| `autograd/engine.h` | `torch::autograd::Engine` |
| `autograd/function.h` | `torch::autograd::Node` |
| `autograd/variable.h` | `torch::autograd::AutogradMeta` |
| `autograd/py_function.h` | `torch::autograd::PyNode` |
| `torch/autograd_engine.py` | `torch/autograd/` + engine |
| `torch/nn/module.py` | `torch/nn/modules/module.py` |
| `torch/nn/transformer.py` | `torch/nn/modules/transformer.py` |
| `torch/optim/adam.py` | `torch/optim/adam.py` |
