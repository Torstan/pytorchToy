"""
最小 AOTAutograd 兼容层。

当前实现先建立 2.x 风格的 fw/bw 编译边界：
- inference: 走 inference_compiler / fw_compiler
- training: 额外准备一个最小 backward graph 并调用 bw_compiler

这一版还没有 joint graph capture，也没有真实的 backward graph lowering。
training 正确性依赖于编译后的 forward 仍然保留 eager autograd 语义。
"""

from dataclasses import dataclass

from torch._compile.graph import Graph, GraphModule
from torch._compile.tracer import Tracer


@dataclass
class AOTCompileState:
    graph_module: object
    backward_graph_module: object
    compiled_fw: object
    compiled_bw: object
    requires_grad: bool


def _value_signature(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        dtype = getattr(getattr(value, "_dtype", None), "name", None)
        return (
            "tensor",
            tuple(value.shape),
            dtype,
            value.requires_grad,
            value.is_contiguous(),
        )
    if isinstance(value, (int, float, str, bool, type(None))):
        return (type(value).__name__, value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_value_signature(item) for item in value))
    if isinstance(value, dict):
        items = tuple((key, _value_signature(item)) for key, item in sorted(value.items()))
        return ("dict", items)
    return (type(value).__name__, id(value))


def _call_signature(args, kwargs):
    return (
        tuple(_value_signature(arg) for arg in args),
        tuple((key, _value_signature(value)) for key, value in sorted(kwargs.items())),
    )


def _clone_structure(value):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_structure(item) for item in value)
    if isinstance(value, list):
        return [_clone_structure(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_structure(item) for key, item in value.items()}
    return value


def _collect_tensor_versions(value, versions):
    from torch.tensor import Tensor

    if isinstance(value, Tensor):
        versions[id(value)] = value._version
        return
    if isinstance(value, tuple):
        for item in value:
            _collect_tensor_versions(item, versions)
        return
    if isinstance(value, list):
        for item in value:
            _collect_tensor_versions(item, versions)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_tensor_versions(item, versions)


def _any_requires_grad(args, kwargs):
    from torch.tensor import Tensor

    def visit(value):
        if isinstance(value, Tensor):
            return value.requires_grad
        if isinstance(value, tuple):
            return any(visit(item) for item in value)
        if isinstance(value, list):
            return any(visit(item) for item in value)
        if isinstance(value, dict):
            return any(visit(item) for item in value.values())
        return False

    return visit(args) or visit(kwargs)


def _assert_no_input_mutation(fn, args, kwargs):
    cloned_args = tuple(_clone_structure(arg) for arg in args)
    cloned_kwargs = {key: _clone_structure(value) for key, value in kwargs.items()}
    before = {}
    _collect_tensor_versions(cloned_args, before)
    _collect_tensor_versions(cloned_kwargs, before)
    fn(*cloned_args, **cloned_kwargs)
    after = {}
    _collect_tensor_versions(cloned_args, after)
    _collect_tensor_versions(cloned_kwargs, after)
    if before != after:
        raise NotImplementedError(
            "AOTAutograd-mini does not support input mutation yet"
        )


def _build_backward_stub_graph():
    graph = Graph()
    grad_output = graph.placeholder("grad_output")
    graph.output(grad_output)
    return GraphModule(graph)


class AOTFunction:
    def __init__(
        self,
        fn,
        *,
        fw_compiler,
        bw_compiler=None,
        inference_compiler=None,
    ):
        self._fn = fn
        self._fw_compiler = fw_compiler
        self._bw_compiler = bw_compiler or fw_compiler
        self._inference_compiler = inference_compiler or fw_compiler
        self._cache = {}
        self._last_state = None

    def __call__(self, *args, **kwargs):
        key = _call_signature(args, kwargs)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = self._compile(args, kwargs)
            self._cache[key] = compiled
        return compiled(*args, **kwargs)

    def _compile(self, args, kwargs):
        if kwargs:
            raise NotImplementedError("AOTAutograd-mini does not support kwargs yet")

        _assert_no_input_mutation(self._fn, args, kwargs)

        tracer = Tracer()
        graph_module = tracer.trace(self._fn, args)
        requires_grad = _any_requires_grad(args, kwargs)

        if requires_grad:
            compiled_fw = self._fw_compiler(graph_module, list(args))
            backward_graph_module = _build_backward_stub_graph()
            compiled_bw = self._bw_compiler(backward_graph_module, [])
        else:
            compiled_fw = self._inference_compiler(graph_module, list(args))
            backward_graph_module = None
            compiled_bw = None

        self._last_state = AOTCompileState(
            graph_module=graph_module,
            backward_graph_module=backward_graph_module,
            compiled_fw=compiled_fw,
            compiled_bw=compiled_bw,
            requires_grad=requires_grad,
        )
        return compiled_fw


def aot_function(
    fn,
    *,
    fw_compiler,
    bw_compiler=None,
    inference_compiler=None,
):
    return AOTFunction(
        fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
    )


def aot_module_simplified(
    fn,
    example_inputs,
    *,
    fw_compiler,
    bw_compiler=None,
    inference_compiler=None,
):
    compiled = aot_function(
        fn,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
    )
    compiled(*example_inputs)
    return compiled


def make_boxed_func(fn):
    return fn
