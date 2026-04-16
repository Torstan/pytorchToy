import torch

from torch._functorch.aot_autograd import aot_function


def eager_compiler(gm, example_inputs):
    del example_inputs

    def compiled(*args):
        return gm(*args)

    return compiled


def fn(x):
    return torch.relu(torch.tanh(x)).sum()


compiled = aot_function(
    fn,
    fw_compiler=eager_compiler,
    bw_compiler=eager_compiler,
)

x = torch.tensor([1.0, -2.0])
x.requires_grad = True

out = compiled(x)
out.backward()

state = compiled._last_state
assert state is not None
assert state.graph_module is not None
assert state.backward_graph_module is not None
assert state.compiled_fw is not None
assert state.compiled_bw is not None

bw_nodes = [node.op for node in state.backward_graph_module.graph.nodes]
assert bw_nodes == ["placeholder", "output"], bw_nodes

print("aot_autograd_fw_bw_partition: ok")
