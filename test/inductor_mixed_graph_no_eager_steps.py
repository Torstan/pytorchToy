import torch

from torch._compile.graph import Node
from torch._compile.pointwise import CompiledGraph, CompiledOpStep, compile_graph_module
from torch._compile.tracer import Tracer
from torch._functorch.aot_autograd import aot_function


def eager_compiler(gm, example_inputs):
    del example_inputs

    def compiled(*args):
        return gm(*args)

    return compiled


def fn(x, w, b):
    return torch.relu(x.mm(w) + b).sum()


x = torch.randn(4, 3)
w = torch.randn(3, 5)
b = torch.randn(5)

fw_graph = Tracer().trace(fn, (x, w, b))
compiled_fw = compile_graph_module(fw_graph, [x, w, b])

assert isinstance(compiled_fw, CompiledGraph)
assert not any(isinstance(step, Node) for step in compiled_fw.steps), compiled_fw.steps
assert any(
    isinstance(step, CompiledOpStep) and step.target == "mm"
    for step in compiled_fw.steps
), compiled_fw.steps
assert any(
    isinstance(step, CompiledOpStep) and step.target == "sum"
    for step in compiled_fw.steps
), compiled_fw.steps
torch.testing.assert_close(compiled_fw.run((x, w, b)), fn(x, w, b))


compiled = aot_function(
    fn,
    fw_compiler=eager_compiler,
    bw_compiler=eager_compiler,
)

x_train = torch.randn(4, 3)
w_train = torch.randn(3, 5)
b_train = torch.randn(5)
x_train.requires_grad = True
w_train.requires_grad = True
b_train.requires_grad = True

x_ref = x_train.clone()
w_ref = w_train.clone()
b_ref = b_train.clone()
x_ref.requires_grad = True
w_ref.requires_grad = True
b_ref.requires_grad = True

ref = fn(x_ref, w_ref, b_ref)
ref.backward()

compiled(x_train, w_train, b_train)
state = compiled._last_state
assert state is not None
compiled_bw = compile_graph_module(
    state.backward_graph_module,
    state.backward_example_inputs,
    allow_requires_grad=True,
)

assert isinstance(compiled_bw, CompiledGraph)
assert not any(isinstance(step, Node) for step in compiled_bw.steps), compiled_bw.steps
assert any(
    isinstance(step, CompiledOpStep) and step.target == "sum"
    for step in compiled_bw.steps
), compiled_bw.steps
assert any(
    isinstance(step, CompiledOpStep) and step.target == "mm"
    for step in compiled_bw.steps
), compiled_bw.steps

bw_out = compiled_bw.run(tuple(state.backward_example_inputs))
assert isinstance(bw_out, tuple)
assert len(bw_out) == 3
torch.testing.assert_close(bw_out[0], x_ref.grad)
torch.testing.assert_close(bw_out[1], w_ref.grad)
torch.testing.assert_close(bw_out[2], b_ref.grad)

print("inductor_mixed_graph_no_eager_steps: ok")
