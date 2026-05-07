# compile2.0 Python Code Organization Design

Date: 2026-05-07
Branch: `compile2.0`

## Context

`pytorchToy` already has a working toy PyTorch 2.x compile pipeline:

`torch.compile -> torch._dynamo.optimize -> graph capture -> GraphModule -> torch._inductor.compile_fx -> decomposition -> AOTAutograd when needed -> lowering/runtime callable`

The current baseline is green:

`python3 test/run_all.py` passed `89/89` tests before design work.

The first refactor round is intentionally limited to the Python compile stack. It will not reorganize C++ code, build artifacts, docs, or the test directory layout. Internal test imports may change, but user-facing compile APIs must remain stable.

## Goals

- Make module names match responsibilities.
- Reduce duplicated helpers and operator metadata.
- Split large mixed-responsibility modules into smaller units.
- Keep compile behavior unchanged.
- Keep public entry points stable:
  - `torch.compile`
  - `torch._dynamo.optimize`
  - `torch._dynamo.reset`
  - `torch._dynamo.list_backends`
  - `torch._inductor.compile_fx`
  - `torch.fx.GraphModule`

## Non-Goals

- No C++ source refactor in this round.
- No test directory reorganization in this round.
- No README or architecture document migration in this round.
- No build output or Makefile layout changes in this round.
- No semantic rewrite of Dynamo, AOTAutograd, PrimTorch, or Inductor behavior.
- No new dynamic shape, frame-eval hook, alias, mutation, or backend capability.

## Recommended Approach

Use a staged "move responsibilities, preserve behavior" refactor.

The implementation should add target modules, migrate imports and internal tests, then remove old duplicated code. Because internal test imports are allowed to change, no compatibility shim is required for old paths such as `torch._compile.pointwise`.

## Target Architecture

### `torch.fx`

Owns the minimal FX layer:

- `torch/fx/graph.py`
  - `Node`
  - `Graph`
  - `GraphModule`
  - graph formatting helpers
  - graph interpretation helpers still needed by tests/runtime
- `torch/fx/proxy.py`
  - `Proxy`
  - `Tracer`
  - tracing state
  - `UnsupportedTraceError`
- `torch/fx/meta.py`
  - meta propagation
  - meta rules using shared operator helpers

`torch.fx.GraphModule.print_readable()` should preserve current output shape as much as practical because tests inspect graph text.

### Shared Operator Helpers

Add one shared helper module, preferably `torch/_compile/ops.py`, for compile-stack operator facts:

- target normalization, replacing repeated `_target_name`
- shape argument normalization, replacing repeated `_normalize_shape_args`
- eager graph op table
- unary/binary/pointwise support sets
- broadcast shape helper
- shared target lists used by Dynamo, FX meta, decomposition, and Inductor lowering

This reduces duplication now spread across:

- `torch/_compile/graph.py`
- `torch/_compile/pointwise.py`
- `torch/_dynamo/symbolic_convert.py`
- `torch/_inductor/decomposition.py`
- `torch/_functorch/aot_autograd.py`
- `torch/fx/meta.py`
- `torch/_subclasses/fake_tensor.py`

### `torch._dynamo`

Keeps these responsibilities:

- `optimize`
- cache and recompile control
- guard generation and matching
- bytecode symbolic capture
- graph-break resume logic
- backend lookup via the stable public API

It should import FX objects from `torch.fx`, not from `torch._compile.graph` or `torch._compile.tracer`.

### `torch._inductor`

Owns compile orchestration and lowering:

- `torch/_inductor/compile_fx.py`
  - public `compile_fx` entry point remains stable
  - orchestration only
- `torch/_inductor/decomposition.py`
  - decomposition table selection and graph rewrite
- `torch/_inductor/lowering/pointwise.py`
  - `PointwiseProgram`
  - pointwise IR
  - native pointwise JIT
  - interpreter-backed pointwise kernel
- `torch/_inductor/lowering/partition.py`
  - mixed graph partitioning
  - `CompiledGraph`
  - `CompiledRegion`
- `torch/_inductor/lowering/single_op.py`
  - runtime kernels for `mm`, `addmm`, `sum`, `t`, `view`, `reshape`, `layer_norm`, `gt`, and simple unary/binary single-node fallbacks

Tests that currently import internals from `torch._compile.pointwise` should import the new Inductor lowering modules instead.

### `torch._functorch`

Keep `torch._functorch.aot_autograd` as the public facade for:

- `aot_function`
- `aot_module_simplified`
- `make_boxed_func`

Move implementation details into `torch/_functorch/_aot_autograd/`:

- `backward_graph.py`
  - backward graph synthesis
  - unsupported backward checks
  - gradient shape reduction helpers
- `runtime.py`
  - compiled backward runtime attachment
  - backward result unwrapping
- `module_lift.py`
  - parameter/buffer lifting
  - module wrapper
  - module tensor swapping
- `utils.py`
  - call signatures
  - value signatures
  - mutation checks

The facade keeps existing import paths stable for user-level and existing AOT tests that import `aot_function` or `aot_module_simplified`.

### `torch._compile`

After the refactor, `_compile` should primarily hold the `torch.compile` API wrapper and shared compile-stack helpers.

It should no longer own:

- FX graph classes
- tracer/proxy classes
- Inductor lowering
- compiled graph runtime

## Data Flow

The data flow must stay behaviorally identical:

1. `torch.compile` delegates to `torch._dynamo.optimize`.
2. Dynamo captures a callable into `torch.fx.GraphModule`.
3. Dynamo resolves the backend and calls it with `(graph_module, example_inputs)`.
4. `torch._inductor.compile_fx` decomposes the graph and propagates metadata.
5. If training inputs require grad, Inductor routes through AOTAutograd.
6. Lowering returns a callable or compiled graph runtime.
7. Unsupported paths keep the existing fallback/error behavior.

## Error Handling

Do not change current error semantics:

- tracing and fullgraph failures still use `UnsupportedTraceError`
- unsupported lowering still uses `PointwiseLoweringError`
- training backward graph gaps still fall back to eager autograd where they do today
- runtime shape/input mismatches should preserve current exception style where practical

## Migration Plan

1. Move FX ownership.
   - Create `torch/fx/graph.py` and expand `torch/fx/proxy.py`.
   - Update Dynamo, AOTAutograd, Inductor, and tests to import from `torch.fx`.
   - Verify `fx_*`, `dynamo_bytecode_*`, and compile legacy tests.

2. Add shared operator helpers.
   - Centralize target names, shape normalization, eager op dispatch, broadcast shape, and op support sets.
   - Update FX meta, decomposition, symbolic convert, fake tensor, and lowering.
   - Verify prims, FX meta, and Inductor tests.

3. Split Inductor lowering.
   - Move pointwise IR/native JIT/runtime into `torch._inductor.lowering.pointwise`.
   - Move partitioned graph runtime into `torch._inductor.lowering.partition`.
   - Move single-op kernels into `torch._inductor.lowering.single_op`.
   - Update tests to use new paths.
   - Verify partition, mixed graph, and training Inductor tests.

4. Split AOTAutograd internals.
   - Keep `torch._functorch.aot_autograd` as facade.
   - Move backward graph, runtime attachment, module lifting, and utilities into `_aot_autograd`.
   - Verify `aot_autograd_*` and compile training tests.

5. Clean `_compile`.
   - Remove old graph/tracer/lowering implementation modules once imports are migrated.
   - Keep `torch.compile` behavior unchanged.
   - Run `python3 test/run_all.py`.

## Verification

Run targeted tests after each migration slice. Before claiming completion, run:

`python3 test/run_all.py`

The expected success condition remains zero failures.

## Risks

- Tests inspect internal classes such as `CompiledGraph` and `CompiledRegion`; update imports and assertions intentionally.
- Graph text is inspected by tests; formatting should not drift unless tests are updated with a clear reason.
- Centralizing op metadata can accidentally change callable target identity in decomposition tests; preserve existing target identity where required.
- AOTAutograd splitting can introduce import cycles; keep the facade thin and move shared utilities to leaf modules.
- Removing old `_compile` modules before all imports are migrated can break hidden dependencies; use `rg` before deletion.

## Open Decisions Resolved

- First refactor scope: Python compile stack only.
- Internal test imports may change.
- Public compile APIs listed in Goals remain stable.
- No old `_compile.pointwise` compatibility shim is required.
