# compile2.0 Python Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the Python `torch.compile` stack so FX, Dynamo, Inductor lowering, and AOTAutograd internals live in modules named for their responsibilities while preserving behavior.

**Architecture:** Move the existing code in small mechanical slices: FX ownership first, shared operator helpers second, Inductor lowering third, AOTAutograd internals fourth, then remove stale `_compile` implementation modules. Public entry points remain stable; internal tests may import new module paths.

**Tech Stack:** Python 3, local script-style tests under `test/`, pybind-backed `_C` and `_nn_C` already built, git for reviewable checkpoints.

---

## Baseline

The design spec is:

`docs/superpowers/specs/2026-05-07-compile2-python-organization-design.md`

Before implementation, the latest known green baseline was:

```bash
python3 test/run_all.py
```

Expected summary:

```text
Passed: 89
Failed: 0
Total : 89
```

Current unrelated untracked files must remain untouched:

```text
ARCHITECTURE.md
ARCHITECTURE_GUIDE.md
ARCHITECTURE_QUICKSTART.md
ARCHITECTURE_SEQUENCE.md
plan_torch_compile
torch_compile_2x_minimal_execution_plan.md
torch_compile_core_modules_analysis.md
torch_compile_pt2_graph_alignment_plan.md
```

## Target File Structure

Create or modify these Python files:

```text
torch/_compile/__init__.py              public torch.compile API wrapper
torch/_compile/ops.py                   shared compile-stack operator helpers
torch/_dynamo/backends/registry.py      backend registry implementation and public Dynamo backend helpers
torch/_dynamo/convert_frame.py          imports UnsupportedTraceError from torch.fx
torch/_dynamo/eval_frame.py             imports Tracer and UnsupportedTraceError from torch.fx
torch/_dynamo/resume_execution.py       imports Tracer and UnsupportedTraceError from torch.fx
torch/_dynamo/symbolic_convert.py       imports Graph/GraphModule and shared op helper sets
torch/_functorch/aot_autograd.py        public facade
torch/_functorch/_aot_autograd/__init__.py
torch/_functorch/_aot_autograd/api.py
torch/_functorch/_aot_autograd/backward_graph.py
torch/_functorch/_aot_autograd/module_lift.py
torch/_functorch/_aot_autograd/runtime.py
torch/_functorch/_aot_autograd/utils.py
torch/_inductor/compile_fx.py           imports compile_graph_module from Inductor lowering
torch/_inductor/decomposition.py        imports FX classes and target_name helper
torch/_inductor/lowering/__init__.py
torch/_inductor/lowering/partition.py
torch/_inductor/lowering/pointwise.py
torch/_inductor/lowering/single_op.py
torch/_subclasses/fake_tensor.py        uses shared broadcast helper
torch/fx/__init__.py                    public minimal FX exports
torch/fx/graph.py                       Node, Graph, GraphModule, graph execution helpers
torch/fx/meta.py                        meta propagation using shared helpers
torch/fx/proxy.py                       Proxy, Tracer, tracing state, UnsupportedTraceError
torch/nn/functional.py                  imports tracing helpers from torch.fx
torch/nn/linear.py                      imports tracing helper from torch.fx
```

Delete these stale implementation files after imports are migrated:

```text
torch/_compile/backend.py
torch/_compile/graph.py
torch/_compile/pointwise.py
torch/_compile/tracer.py
```

Internal tests to update:

```text
test/compile_fullgraph_error.py
test/compile_legacy_path_still_works.py
test/compile_partitioned_graph.py
test/dynamo_bytecode_capture_basic.py
test/dynamo_bytecode_compare_branch.py
test/dynamo_bytecode_static_branch_bool.py
test/dynamo_call_inline_basic.py
test/dynamo_constant_and_tensor_mixed.py
test/inductor_broadcast_pointwise_fused.py
test/inductor_compile_fx_basic.py
test/inductor_mixed_graph_no_eager_steps.py
test/prims_backend_contract.py
test/prims_decomposition_basic.py
test/prims_layer_norm_lowering_input.py
test/prims_linear_lowering_input.py
```

---

## Task 1: Move FX Graph and Tracer Ownership

**Files:**
- Create: `torch/fx/graph.py`
- Modify: `torch/fx/proxy.py`
- Modify: `torch/fx/__init__.py`
- Modify: `torch/_dynamo/convert_frame.py`
- Modify: `torch/_dynamo/eval_frame.py`
- Modify: `torch/_dynamo/resume_execution.py`
- Modify: `torch/_dynamo/symbolic_convert.py`
- Modify: `torch/_functorch/aot_autograd.py`
- Modify: `torch/_inductor/decomposition.py`
- Modify: `torch/nn/functional.py`
- Modify: `torch/nn/linear.py`
- Modify tracer-importing tests listed above

- [ ] **Step 1: Update one test import to define the new tracer path**

Edit `test/compile_fullgraph_error.py`:

```python
import torch
from torch.fx import UnsupportedTraceError


def demo(x):
    if x.sum().item() > 0:
        return x + 1
    return x - 1


compiled = torch.compile(demo, backend="eager", fullgraph=True)
x = torch.ones([2, 2])
try:
    compiled(x)
    raise AssertionError("expected UnsupportedTraceError")
except UnsupportedTraceError:
    pass
```

- [ ] **Step 2: Run the updated test and verify the new path is missing**

Run:

```bash
python3 test/compile_fullgraph_error.py
```

Expected before implementation:

```text
ImportError: cannot import name 'UnsupportedTraceError' from 'torch.fx'
```

- [ ] **Step 3: Create `torch/fx/graph.py` from the current graph implementation**

Move the current contents of `torch/_compile/graph.py` into `torch/fx/graph.py`.

Keep these public names in `torch/fx/graph.py`:

```python
Node
Graph
GraphModule
target_name
normalize_shape_args
register_eager_op
EAGER_OP_TABLE
interpret
resolve
```

Use these names as renames from the old private helpers:

```python
_target_name -> target_name
_normalize_shape_args -> normalize_shape_args
_register_op -> register_eager_op
_OP_TABLE -> EAGER_OP_TABLE
_interpret -> interpret
_resolve -> resolve
```

Inside `GraphModule.forward`, call `interpret(self.graph, args)`.

Inside `GraphModule.propagate_meta`, keep the lazy import:

```python
from torch.fx.meta import propagate_meta
```

- [ ] **Step 4: Move tracer implementation into `torch/fx/proxy.py`**

Replace the current wrapper in `torch/fx/proxy.py` with the current contents of `torch/_compile/tracer.py`.

Update the top import in the moved code:

```python
import inspect
import threading

from torch.fx.graph import Graph, GraphModule, Node
```

Keep these public names in `torch/fx/proxy.py`:

```python
UnsupportedTraceError
current_tracer
is_tracing
Proxy
Tracer
```

- [ ] **Step 5: Export FX public names from `torch/fx/__init__.py`**

Replace `torch/fx/__init__.py` with:

```python
"""
最小 torch.fx 兼容层。
"""

from .graph import Graph, GraphModule, Node
from .meta import propagate_meta
from .proxy import (
    Proxy,
    Tracer,
    UnsupportedTraceError,
    current_tracer,
    is_tracing,
)

__all__ = [
    "Graph",
    "GraphModule",
    "Node",
    "Proxy",
    "Tracer",
    "UnsupportedTraceError",
    "current_tracer",
    "is_tracing",
    "propagate_meta",
]
```

- [ ] **Step 6: Update production imports to use `torch.fx`**

Apply these exact import rewrites:

```python
# torch/_dynamo/convert_frame.py
from torch.fx import UnsupportedTraceError

# torch/_dynamo/eval_frame.py
from torch.fx import Tracer, UnsupportedTraceError

# torch/_dynamo/resume_execution.py
from torch.fx import Tracer, UnsupportedTraceError

# torch/_dynamo/symbolic_convert.py
from torch.fx import Graph, GraphModule, UnsupportedTraceError

# torch/_functorch/aot_autograd.py
from torch.fx import Graph, GraphModule, Node, Tracer

# torch/_inductor/decomposition.py
from torch.fx import Graph, GraphModule, Node

# torch/nn/functional.py
from torch.fx import current_tracer, is_tracing

# torch/nn/linear.py
from torch.fx import is_tracing
```

For local imports inside functions, use the same names from `torch.fx`.

- [ ] **Step 7: Update tests that import `Tracer` or tracer module**

Use these import forms:

```python
from torch.fx import Tracer
```

For tests that monkey-patch the tracer module, use:

```python
import torch.fx.proxy as tracer_mod
```

Update these files:

```text
test/compile_legacy_path_still_works.py
test/dynamo_bytecode_capture_basic.py
test/dynamo_bytecode_compare_branch.py
test/dynamo_bytecode_static_branch_bool.py
test/dynamo_call_inline_basic.py
test/dynamo_constant_and_tensor_mixed.py
test/inductor_broadcast_pointwise_fused.py
test/inductor_compile_fx_basic.py
test/inductor_mixed_graph_no_eager_steps.py
test/prims_backend_contract.py
test/prims_decomposition_basic.py
test/prims_layer_norm_lowering_input.py
test/prims_linear_lowering_input.py
```

- [ ] **Step 8: Verify no production code imports FX from `_compile`**

Run:

```bash
rg -n "from torch\\._compile\\.(graph|tracer)|import torch\\._compile\\.(graph|tracer)" torch test
```

Expected output after this task:

```text
torch/_compile/tracer.py:16:from torch._compile.graph import Graph, GraphModule, Node
```

The remaining match is acceptable only until the stale file is removed in the cleanup task.

- [ ] **Step 9: Run targeted tests**

Run:

```bash
python3 test/fx_graph_basic.py
python3 test/fx_graph_meta_prop.py
python3 test/compile_fullgraph_error.py
python3 test/dynamo_bytecode_capture_basic.py
python3 test/compile_legacy_path_still_works.py
```

Expected: all commands exit with status `0`.

- [ ] **Step 10: Commit Task 1**

Run:

```bash
git add torch/fx torch/_dynamo torch/_functorch/aot_autograd.py torch/_inductor/decomposition.py torch/nn/functional.py torch/nn/linear.py test
git commit -m "refactor: move fx graph and tracer ownership"
```

---

## Task 2: Centralize Compile Operator Helpers

**Files:**
- Create: `torch/_compile/ops.py`
- Modify: `torch/fx/graph.py`
- Modify: `torch/fx/meta.py`
- Modify: `torch/_subclasses/fake_tensor.py`
- Modify: `torch/_dynamo/symbolic_convert.py`
- Modify: `torch/_inductor/decomposition.py`

- [ ] **Step 1: Add a direct import test for shared helpers**

Create `test/compile_ops_helpers.py`:

```python
from torch._compile.ops import (
    BINARY_POINTWISE_TARGETS,
    TENSOR_METHOD_NAMES,
    TORCH_OPERATOR_NAMES,
    UNARY_POINTWISE_TARGETS,
    broadcast_shapes,
    normalize_shape_args,
    target_name,
)


def sample():
    pass


assert target_name("add") == "add"
assert target_name(sample) == "sample"
assert normalize_shape_args(("x", (2, 3))) == (2, 3)
assert normalize_shape_args(("x", 2, 3)) == (2, 3)
assert broadcast_shapes((2, 1), (1, 3)) == (2, 3)
assert "sin" in UNARY_POINTWISE_TARGETS
assert "add" in BINARY_POINTWISE_TARGETS
assert "relu" in TORCH_OPERATOR_NAMES
assert "view" in TENSOR_METHOD_NAMES
```

- [ ] **Step 2: Run the helper test and verify it fails before the module exists**

Run:

```bash
python3 test/compile_ops_helpers.py
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'torch._compile.ops'
```

- [ ] **Step 3: Create `torch/_compile/ops.py`**

Add:

```python
"""
Shared operator helpers for the toy compile stack.
"""

UNARY_POINTWISE_TARGETS = frozenset({"sin", "cos", "relu", "tanh", "neg"})
BINARY_POINTWISE_TARGETS = frozenset({"add", "sub", "mul", "div"})
POINTWISE_TARGETS = UNARY_POINTWISE_TARGETS | BINARY_POINTWISE_TARGETS

TORCH_OPERATOR_NAMES = frozenset({
    "sin",
    "cos",
    "exp",
    "log",
    "relu",
    "tanh",
    "sum",
    "view",
    "reshape",
    "mm",
    "addmm",
})

TENSOR_METHOD_NAMES = frozenset({
    "sin",
    "cos",
    "exp",
    "log",
    "relu",
    "tanh",
    "sum",
    "view",
    "reshape",
    "mm",
    "t",
    "gt",
})


def target_name(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        return target.__name__
    return repr(target)


def normalize_shape_args(args):
    if len(args) == 2 and isinstance(args[1], (tuple, list)):
        return tuple(args[1])
    return tuple(args[1:])


def broadcast_shapes(lhs_shape, rhs_shape, *, error_type=RuntimeError):
    lhs = list(lhs_shape)
    rhs = list(rhs_shape)
    result = []
    while lhs or rhs:
        left = lhs.pop() if lhs else 1
        right = rhs.pop() if rhs else 1
        if left == 1:
            result.append(right)
            continue
        if right == 1 or left == right:
            result.append(left)
            continue
        raise error_type(f"cannot broadcast shapes {lhs_shape} and {rhs_shape}")
    result.reverse()
    return tuple(result)


EAGER_OP_TABLE = {}


def register_eager_op(name):
    def decorator(fn):
        EAGER_OP_TABLE[name] = fn
        return fn
    return decorator


@register_eager_op("sin")
def _op_sin(args, kwargs):
    del kwargs
    return args[0].sin()


@register_eager_op("cos")
def _op_cos(args, kwargs):
    del kwargs
    return args[0].cos()


@register_eager_op("exp")
def _op_exp(args, kwargs):
    del kwargs
    return args[0].exp()


@register_eager_op("log")
def _op_log(args, kwargs):
    del kwargs
    return args[0].log()


@register_eager_op("add")
def _op_add(args, kwargs):
    del kwargs
    return args[0] + args[1]


@register_eager_op("sub")
def _op_sub(args, kwargs):
    del kwargs
    return args[0] - args[1]


@register_eager_op("mul")
def _op_mul(args, kwargs):
    del kwargs
    return args[0] * args[1]


@register_eager_op("div")
def _op_div(args, kwargs):
    del kwargs
    return args[0] / args[1]


@register_eager_op("neg")
def _op_neg(args, kwargs):
    del kwargs
    return -args[0]


@register_eager_op("relu")
def _op_relu(args, kwargs):
    del kwargs
    return args[0].relu()


@register_eager_op("tanh")
def _op_tanh(args, kwargs):
    del kwargs
    return args[0].tanh()


@register_eager_op("sum")
def _op_sum(args, kwargs):
    return args[0].sum(**kwargs)


@register_eager_op("gt")
def _op_gt(args, kwargs):
    del kwargs
    return args[0].gt(args[1])


@register_eager_op("t")
def _op_t(args, kwargs):
    del kwargs
    return args[0].t()


@register_eager_op("mm")
def _op_mm(args, kwargs):
    del kwargs
    return args[0].mm(args[1])


@register_eager_op("view")
def _op_view(args, kwargs):
    del kwargs
    return args[0].view(*normalize_shape_args(args))


@register_eager_op("reshape")
def _op_reshape(args, kwargs):
    del kwargs
    return args[0].reshape(*normalize_shape_args(args))


@register_eager_op("addmm")
def _op_addmm(args, kwargs):
    del kwargs
    bias, lhs, rhs = args
    return lhs.mm(rhs) + bias


@register_eager_op("layer_norm")
def _op_layer_norm(args, kwargs):
    import torch.nn.functional as F

    return F.layer_norm(*args, **kwargs)


@register_eager_op("call_callable")
def _op_call_callable(args, kwargs):
    return args[0](*args[1:], **kwargs)


def run_eager_target(target, args, kwargs):
    op_fn = EAGER_OP_TABLE.get(target)
    if op_fn is not None:
        return op_fn(args, kwargs)
    if not isinstance(target, str):
        return target(*args, **kwargs)
    raise RuntimeError(f"unsupported compiled target: {target}")
```

- [ ] **Step 4: Update `torch/fx/graph.py` to use shared ops**

In `torch/fx/graph.py`, import:

```python
from torch._compile.ops import (
    EAGER_OP_TABLE,
    normalize_shape_args,
    register_eager_op,
    run_eager_target,
    target_name,
)
```

Remove the local eager op table and local target/shape helpers. In `Graph.call_function`, name nodes with `target_name(target)`. In `interpret`, replace manual op lookup with:

```python
env[node.name] = run_eager_target(node.target, call_args, call_kwargs)
```

Keep exported aliases only if production code still imports them during this task:

```python
_target_name = target_name
_normalize_shape_args = normalize_shape_args
_OP_TABLE = EAGER_OP_TABLE
_register_op = register_eager_op
```

These aliases are removed in the cleanup task after all references are migrated.

- [ ] **Step 5: Update `torch/fx/meta.py`**

Replace local `_broadcast_shapes` and `_target_name` usage with:

```python
from torch._compile.ops import broadcast_shapes, target_name
```

In `_binary_meta` and `_addmm_meta`, call:

```python
shape = broadcast_shapes(lhs.shape, rhs.shape, error_type=MetaPropagationError)
```

In `infer_meta`, use:

```python
name = target_name(target)
```

- [ ] **Step 6: Update `torch/_subclasses/fake_tensor.py`**

Replace local `_broadcast_shapes` with:

```python
from torch._compile.ops import broadcast_shapes
```

In `FakeTensor._binary`, call:

```python
broadcast_shapes(self.shape, tuple(other.shape))
```

- [ ] **Step 7: Update Dynamo symbolic operator sets**

In `torch/_dynamo/symbolic_convert.py`, replace local operator sets with:

```python
from torch._compile.ops import TENSOR_METHOD_NAMES, TORCH_OPERATOR_NAMES
```

Update references:

```python
_TENSOR_METHOD_NAMES -> TENSOR_METHOD_NAMES
_TORCH_OPERATOR_NAMES -> TORCH_OPERATOR_NAMES
```

- [ ] **Step 8: Update decomposition target helper**

In `torch/_inductor/decomposition.py`, import:

```python
from torch._compile.ops import target_name
```

Replace `_target_name(node.target)` with `target_name(node.target)` and remove the local helper.

- [ ] **Step 9: Run helper and targeted tests**

Run:

```bash
python3 test/compile_ops_helpers.py
python3 test/fx_graph_meta_prop.py
python3 test/fake_tensor_meta_basic.py
python3 test/prims_decomposition_basic.py
python3 test/prims_backend_contract.py
```

Expected: all commands exit with status `0`.

- [ ] **Step 10: Commit Task 2**

Run:

```bash
git add torch/_compile/ops.py torch/fx/graph.py torch/fx/meta.py torch/_subclasses/fake_tensor.py torch/_dynamo/symbolic_convert.py torch/_inductor/decomposition.py test/compile_ops_helpers.py
git commit -m "refactor: centralize compile operator helpers"
```

---

## Task 3: Split Inductor Lowering Out of `_compile.pointwise`

**Files:**
- Create: `torch/_inductor/lowering/__init__.py`
- Create: `torch/_inductor/lowering/pointwise.py`
- Create: `torch/_inductor/lowering/partition.py`
- Create: `torch/_inductor/lowering/single_op.py`
- Modify: `torch/_inductor/compile_fx.py`
- Modify: `test/compile_legacy_path_still_works.py`
- Modify: `test/compile_partitioned_graph.py`
- Modify: `test/inductor_broadcast_pointwise_fused.py`
- Modify: `test/inductor_mixed_graph_no_eager_steps.py`

- [ ] **Step 1: Update one Inductor test import to define the new lowering path**

Edit `test/compile_partitioned_graph.py` imports:

```python
import torch
from torch.fx import Tracer
from torch._inductor.lowering.partition import (
    CompiledGraph,
    CompiledRegion,
    compile_graph_module,
)
```

Leave the rest of the test body unchanged.

- [ ] **Step 2: Run the updated test and verify the new module is missing**

Run:

```bash
python3 test/compile_partitioned_graph.py
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'torch._inductor.lowering'
```

- [ ] **Step 3: Create `torch/_inductor/lowering/pointwise.py`**

Move these definitions unchanged from `torch/_compile/pointwise.py` into `torch/_inductor/lowering/pointwise.py`:

```python
PointwiseLoweringError
ValueRef
Instruction
PointwiseProgram
NativePointwiseKernel
CppPointwiseKernel
_format_cpp_float
_is_broadcastable_to
_prepare_pointwise_runtime_args
_broadcast_shapes
lower_pointwise_graph
```

At the top of the new file, import shared op facts:

```python
import ctypes
from dataclasses import dataclass
import hashlib
import os
import shutil
import subprocess
import tempfile

from torch._compile.ops import (
    BINARY_POINTWISE_TARGETS,
    POINTWISE_TARGETS,
    UNARY_POINTWISE_TARGETS,
    broadcast_shapes,
    target_name,
)
```

Replace old constants:

```python
_UNARY_TARGETS -> UNARY_POINTWISE_TARGETS
_BINARY_TARGETS -> BINARY_POINTWISE_TARGETS
_SUPPORTED_TARGETS -> POINTWISE_TARGETS
```

Replace the local `_broadcast_shapes` body with a thin wrapper so error text stays lowering-specific:

```python
def _broadcast_shapes(lhs_shape, rhs_shape):
    return broadcast_shapes(
        lhs_shape,
        rhs_shape,
        error_type=PointwiseLoweringError,
    )
```

- [ ] **Step 4: Create `torch/_inductor/lowering/single_op.py`**

Move these definitions unchanged from `torch/_compile/pointwise.py` into `torch/_inductor/lowering/single_op.py`:

```python
InputRef
SingleNodeKernel
UnaryPointwiseKernel
BinaryPointwiseKernel
GtKernel
MmKernel
AddmmKernel
SumKernel
TransposeKernel
ViewKernel
ReshapeKernel
LayerNormKernel
_materialize_input_specs
_run_kernel_target
```

Add `try_compile_single_op` by moving the old `_try_compile_single_op` function, keeping the body, and changing the function declaration from:

```python
def _try_compile_single_op(node, env_example):
```

to:

```python
def try_compile_single_op(node, env_example):
```

At the top of the file, import:

```python
from dataclasses import dataclass

from torch._compile.ops import (
    BINARY_POINTWISE_TARGETS,
    UNARY_POINTWISE_TARGETS,
    normalize_shape_args,
    run_eager_target,
    target_name,
)
from torch._inductor.lowering.pointwise import PointwiseLoweringError
```

In `try_compile_single_op`, replace:

```python
_normalize_shape_args -> normalize_shape_args
_target_name -> target_name
_UNARY_TARGETS -> UNARY_POINTWISE_TARGETS
_BINARY_TARGETS -> BINARY_POINTWISE_TARGETS
```

In `_run_kernel_target`, use:

```python
return run_eager_target(target, call_args, call_kwargs)
```

- [ ] **Step 5: Create `torch/_inductor/lowering/partition.py`**

Move these definitions from `torch/_compile/pointwise.py` into `torch/_inductor/lowering/partition.py`:

```python
CompiledRegion
CompiledOpStep
CompiledGraph
compile_graph_module
_compile_partitioned_graph
_try_compile_region
_build_region_graph_module
_region_has_single_output
_build_users
_run_call_function_node
```

At the top of the file, import:

```python
from dataclasses import dataclass

from torch._compile.ops import POINTWISE_TARGETS, run_eager_target, target_name
from torch._inductor.lowering.pointwise import (
    PointwiseLoweringError,
    lower_pointwise_graph,
)
from torch._inductor.lowering.single_op import try_compile_single_op
```

Replace old references:

```python
_SUPPORTED_TARGETS -> POINTWISE_TARGETS
_target_name -> target_name
_try_compile_single_op -> try_compile_single_op
_OP_TABLE manual lookup -> run_eager_target
```

Use FX imports in local helper functions:

```python
from torch.fx import Graph, GraphModule, Node
from torch.fx.graph import resolve
```

- [ ] **Step 6: Export lowering names**

Create `torch/_inductor/lowering/__init__.py`:

```python
from .partition import (
    CompiledGraph,
    CompiledOpStep,
    CompiledRegion,
    compile_graph_module,
)
from .pointwise import (
    CppPointwiseKernel,
    NativePointwiseKernel,
    PointwiseLoweringError,
    PointwiseProgram,
    lower_pointwise_graph,
)

__all__ = [
    "CompiledGraph",
    "CompiledOpStep",
    "CompiledRegion",
    "compile_graph_module",
    "CppPointwiseKernel",
    "NativePointwiseKernel",
    "PointwiseLoweringError",
    "PointwiseProgram",
    "lower_pointwise_graph",
]
```

- [ ] **Step 7: Update `torch/_inductor/compile_fx.py`**

Replace:

```python
from torch._compile.pointwise import PointwiseLoweringError, compile_graph_module
```

with:

```python
from torch._inductor.lowering.partition import compile_graph_module
from torch._inductor.lowering.pointwise import PointwiseLoweringError
```

- [ ] **Step 8: Update Inductor tests to import new lowering modules**

Use these import rewrites:

```python
# test/compile_legacy_path_still_works.py
from torch._inductor.lowering.partition import compile_graph_module

# test/inductor_broadcast_pointwise_fused.py
from torch._inductor.lowering.pointwise import CppPointwiseKernel, NativePointwiseKernel
from torch._inductor.lowering.partition import compile_graph_module

# test/inductor_mixed_graph_no_eager_steps.py
from torch.fx import Node, Tracer
from torch._inductor.lowering.partition import CompiledGraph, CompiledRegion, compile_graph_module
from torch._inductor.lowering.single_op import SingleNodeKernel
```

If `SingleNodeKernel` remains unused in production after the split, move it with the other single-op kernels and keep the test assertion pointed at the new module.

- [ ] **Step 9: Verify no code imports `_compile.pointwise`**

Run:

```bash
rg -n "torch\\._compile\\.pointwise|from torch\\._compile\\.pointwise|import torch\\._compile\\.pointwise" torch test
```

Expected output:

```text
```

- [ ] **Step 10: Run targeted Inductor tests**

Run:

```bash
python3 test/compile_legacy_path_still_works.py
python3 test/compile_partitioned_graph.py
python3 test/inductor_broadcast_pointwise_fused.py
python3 test/inductor_compile_fx_basic.py
python3 test/inductor_mixed_graph_no_eager_steps.py
python3 test/compile_training_inductor_pointwise_compiles_fw_bw.py
python3 test/compile_training_inductor_pointwise_two_grads.py
```

Expected: all commands exit with status `0`.

- [ ] **Step 11: Commit Task 3**

Run:

```bash
git add torch/_inductor torch/_compile/ops.py test/compile_legacy_path_still_works.py test/compile_partitioned_graph.py test/inductor_broadcast_pointwise_fused.py test/inductor_mixed_graph_no_eager_steps.py
git commit -m "refactor: move inductor lowering modules"
```

---

## Task 4: Split AOTAutograd Internals Behind the Existing Facade

**Files:**
- Create: `torch/_functorch/_aot_autograd/__init__.py`
- Create: `torch/_functorch/_aot_autograd/api.py`
- Create: `torch/_functorch/_aot_autograd/backward_graph.py`
- Create: `torch/_functorch/_aot_autograd/module_lift.py`
- Create: `torch/_functorch/_aot_autograd/runtime.py`
- Create: `torch/_functorch/_aot_autograd/utils.py`
- Modify: `torch/_functorch/aot_autograd.py`

- [ ] **Step 1: Add an internal import smoke test**

Create `test/aot_autograd_internal_modules.py`:

```python
from torch._functorch.aot_autograd import (
    aot_function,
    aot_module_simplified,
    make_boxed_func,
)
from torch._functorch._aot_autograd.backward_graph import build_backward_graph_or_stub
from torch._functorch._aot_autograd.runtime import attach_compiled_backward
from torch._functorch._aot_autograd.utils import call_signature


assert callable(aot_function)
assert callable(aot_module_simplified)
assert make_boxed_func(lambda x: x)(3) == 3
assert callable(build_backward_graph_or_stub)
assert callable(attach_compiled_backward)
assert callable(call_signature)
```

- [ ] **Step 2: Run the smoke test and verify internal modules are missing**

Run:

```bash
python3 test/aot_autograd_internal_modules.py
```

Expected before implementation:

```text
ModuleNotFoundError: No module named 'torch._functorch._aot_autograd'
```

- [ ] **Step 3: Create `utils.py`**

Move these definitions from `torch/_functorch/aot_autograd.py` into `torch/_functorch/_aot_autograd/utils.py` and rename public helper wrappers without leading underscores:

```python
AOTCompileState
_value_signature -> value_signature
_call_signature -> call_signature
_clone_structure -> clone_structure
_collect_tensor_versions -> collect_tensor_versions
_any_requires_grad -> any_requires_grad
_differentiable_input_indices -> differentiable_input_indices
_assert_no_input_mutation -> assert_no_input_mutation
```

Keep compatibility aliases inside `utils.py` for internal migration:

```python
_value_signature = value_signature
_call_signature = call_signature
_clone_structure = clone_structure
_collect_tensor_versions = collect_tensor_versions
_any_requires_grad = any_requires_grad
_differentiable_input_indices = differentiable_input_indices
_assert_no_input_mutation = assert_no_input_mutation
```

- [ ] **Step 4: Create `backward_graph.py`**

Move these definitions from `torch/_functorch/aot_autograd.py` into `torch/_functorch/_aot_autograd/backward_graph.py`:

```python
_build_backward_stub_graph
_target_name
_UnsupportedBackwardGraph
_rebuild_forward_value
_zeros_like_graph_value
_shape_of_node
_reduce_grad_to_shape
_grad_to_target
_build_backward_graph
_build_backward_graph_or_stub
```

Rename the exported entry point:

```python
def build_backward_graph_or_stub(graph_module, example_inputs):
    return _build_backward_graph_or_stub(graph_module, example_inputs)
```

Import shared target helper:

```python
from torch._compile.ops import target_name
```

Use `target_name(node.target)` instead of the local `_target_name` body.

- [ ] **Step 5: Create `runtime.py`**

Move these definitions into `torch/_functorch/_aot_autograd/runtime.py`:

```python
_unwrap_backward_result
_CompiledBackwardRuntime
_attach_compiled_backward
```

Add exported wrapper:

```python
def attach_compiled_backward(output, args, compiled_bw, differentiable_input_indices):
    return _attach_compiled_backward(
        output,
        args,
        compiled_bw,
        differentiable_input_indices,
    )
```

- [ ] **Step 6: Create `api.py`**

Move `AOTFunction`, `aot_function`, and `make_boxed_func` into `torch/_functorch/_aot_autograd/api.py`.

Use these imports at the top:

```python
from torch.fx import Tracer

from .backward_graph import build_backward_graph_or_stub
from .runtime import attach_compiled_backward
from .utils import (
    AOTCompileState,
    any_requires_grad,
    assert_no_input_mutation,
    call_signature,
    differentiable_input_indices,
)
```

In `AOTFunction.__call__`, replace:

```python
key = _call_signature(args, kwargs)
```

with:

```python
key = call_signature(args, kwargs)
```

In `_compile`, replace old helper calls with imported helper names:

```python
assert_no_input_mutation(self._fn, args, kwargs)
requires_grad = any_requires_grad(args, kwargs)
differentiable_input_indices = differentiable_input_indices(args)
backward_graph_module, backward_example_inputs, backward_is_real = build_backward_graph_or_stub(graph_module, list(args))
return attach_compiled_backward(output, runtime_args, compiled_bw, differentiable_input_indices)
```

Use a local variable named `diff_input_indices` to avoid shadowing the imported function:

```python
diff_input_indices = differentiable_input_indices(args)
```

- [ ] **Step 7: Create `module_lift.py`**

Move these definitions into `torch/_functorch/_aot_autograd/module_lift.py`:

```python
aot_module_simplified
_named_buffers
_resolve_module_owner
_swap_module_tensor
_AOTModuleWrapper
```

At the top, import:

```python
from .api import aot_function
```

- [ ] **Step 8: Create `_aot_autograd/__init__.py`**

Add:

```python
from .api import AOTFunction, aot_function, make_boxed_func
from .module_lift import aot_module_simplified

__all__ = [
    "AOTFunction",
    "aot_function",
    "aot_module_simplified",
    "make_boxed_func",
]
```

- [ ] **Step 9: Replace the public facade**

Replace `torch/_functorch/aot_autograd.py` with:

```python
"""
Public AOTAutograd facade.
"""

from torch._functorch._aot_autograd import (
    AOTFunction,
    aot_function,
    aot_module_simplified,
    make_boxed_func,
)

__all__ = [
    "AOTFunction",
    "aot_function",
    "aot_module_simplified",
    "make_boxed_func",
]
```

- [ ] **Step 10: Run AOTAutograd tests**

Run:

```bash
python3 test/aot_autograd_internal_modules.py
python3 test/aot_autograd_inference_basic.py
python3 test/aot_autograd_backward_basic.py
python3 test/aot_autograd_fw_bw_partition.py
python3 test/aot_autograd_runtime_compiled_backward.py
python3 test/aot_autograd_module_params.py
python3 test/aot_autograd_module_params_backward_graph_values.py
python3 test/compile_training_aot_inductor_basic.py
python3 test/compile_training_linear_aot_inductor.py
```

Expected: all commands exit with status `0`.

- [ ] **Step 11: Commit Task 4**

Run:

```bash
git add torch/_functorch test/aot_autograd_internal_modules.py
git commit -m "refactor: split aot autograd internals"
```

---

## Task 5: Move Backend Registry and Remove Stale `_compile` Modules

**Files:**
- Modify: `torch/_dynamo/backends/registry.py`
- Modify: `torch/_dynamo/__init__.py`
- Modify: `torch/_compile/__init__.py`
- Delete: `torch/_compile/backend.py`
- Delete: `torch/_compile/graph.py`
- Delete: `torch/_compile/pointwise.py`
- Delete: `torch/_compile/tracer.py`

- [ ] **Step 1: Make backend registry self-contained**

Replace `torch/_dynamo/backends/registry.py` with:

```python
"""
Minimal Dynamo backend registry.
"""

_BACKENDS = {}


def register_backend(name):
    def decorator(fn):
        _BACKENDS[name] = fn
        return fn
    return decorator


def lookup_backend(backend):
    if callable(backend) and not isinstance(backend, str):
        return backend
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend}. Available: {list(_BACKENDS.keys())}"
            )
        return _BACKENDS[backend]
    raise TypeError(f"backend must be str or callable, got {type(backend)}")


def list_backends():
    return sorted(_BACKENDS.keys())


@register_backend("eager")
def eager_backend(graph_module, example_inputs):
    del example_inputs

    def compiled_fn(*args):
        return graph_module(*args)

    return compiled_fn


@register_backend("inductor")
def inductor_backend(graph_module, example_inputs):
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(graph_module, example_inputs)


@register_backend("default")
def default_backend(graph_module, example_inputs):
    del example_inputs

    def compiled_fn(*args):
        return graph_module(*args)

    return compiled_fn
```

- [ ] **Step 2: Confirm public Dynamo exports still point to the registry**

Ensure `torch/_dynamo/__init__.py` keeps:

```python
from .backends.registry import list_backends, lookup_backend
```

- [ ] **Step 3: Keep `torch/_compile/__init__.py` focused on compile API**

Ensure `torch/_compile/__init__.py` only implements the `compile` function and imports no removed implementation modules.

The file should still call:

```python
decorator = torch._dynamo.optimize(
    backend=backend,
    nopython=fullgraph,
    dynamic=dynamic,
    disable=disable,
)
```

- [ ] **Step 4: Remove stale `_compile` implementation files**

Run:

```bash
git rm torch/_compile/backend.py
git rm torch/_compile/graph.py
git rm torch/_compile/pointwise.py
git rm torch/_compile/tracer.py
```

Keep:

```text
torch/_compile/__init__.py
torch/_compile/ops.py
```

- [ ] **Step 5: Verify removed paths are not imported**

Run:

```bash
rg -n "torch\\._compile\\.(backend|graph|pointwise|tracer)|from torch\\._compile\\.(backend|graph|pointwise|tracer)|import torch\\._compile\\.(backend|graph|pointwise|tracer)" torch test
```

Expected output:

```text
```

- [ ] **Step 6: Run backend and compile API tests**

Run:

```bash
python3 test/dynamo_backend_registry.py
python3 test/compile_routes_through_dynamo.py
python3 test/compile.py
python3 test/test_compile.py
python3 test/test_compile2.py
python3 test/compile_inductor_routes_through_compile_fx.py
```

Expected: all commands exit with status `0`.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add torch/_compile torch/_dynamo test
git commit -m "refactor: clean compile package implementation modules"
```

---

## Task 6: Full Verification and Final Review

**Files:**
- No new source files.
- Review all files changed by Tasks 1 through 5.

- [ ] **Step 1: Run a repository import scan**

Run:

```bash
rg -n "torch\\._compile\\.(backend|graph|pointwise|tracer)|from torch\\._compile\\.(backend|graph|pointwise|tracer)|import torch\\._compile\\.(backend|graph|pointwise|tracer)" torch test
```

Expected output:

```text
```

- [ ] **Step 2: Run duplicate helper scan**

Run:

```bash
rg -n "def _target_name|def _normalize_shape_args|def _broadcast_shapes|_OP_TABLE|_UNARY_TARGETS|_BINARY_TARGETS|_SUPPORTED_TARGETS|_TORCH_OPERATOR_NAMES|_TENSOR_METHOD_NAMES" torch
```

Expected output may include compatibility aliases only in `torch/fx/graph.py` if they were still needed during migration. If no production import needs them, remove those aliases and rerun this command until only intentional names in `torch/_compile/ops.py` remain.

- [ ] **Step 3: Run full test suite**

Run:

```bash
python3 test/run_all.py
```

Expected summary:

```text
Passed: 91
Failed: 0
Total : 91
```

The total is expected to increase from `89` to `91` because this plan adds:

```text
test/compile_ops_helpers.py
test/aot_autograd_internal_modules.py
```

- [ ] **Step 4: Inspect git status**

Run:

```bash
git status --short --branch
```

Expected tracked changes are committed. The pre-existing untracked architecture files may still be listed.

- [ ] **Step 5: Commit final verification note if any cleanup was needed**

If Step 2 required cleanup changes after Task 5, commit them:

```bash
git add torch test
git commit -m "refactor: finish compile python organization cleanup"
```

If Step 2 required no cleanup changes, do not create an empty commit.
