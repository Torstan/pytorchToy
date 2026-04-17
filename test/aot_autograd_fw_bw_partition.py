import torch

from torch._functorch.aot_autograd import aot_function

from test_utils import eager_compiler


def fn(x):
    return torch.relu(torch.tanh(x)).sum()


compiled = aot_function(
    fn,
    fw_compiler=eager_compiler,
    bw_compiler=eager_compiler,
)

x = torch.tensor([1.0, -2.0])
x.requires_grad = True

x_ref = torch.tensor([1.0, -2.0])
x_ref.requires_grad = True

ref = fn(x_ref)
ref.backward()

out = compiled(x)
out.backward()

state = compiled._last_state
assert state is not None
assert state.graph_module is not None
assert state.backward_graph_module is not None
assert state.compiled_fw is not None
assert state.compiled_bw is not None
assert state.backward_is_real is True

bw_out = state.compiled_bw(*state.backward_example_inputs)
assert isinstance(bw_out, tuple)
assert len(bw_out) == 1
torch.testing.assert_close(bw_out[0], x_ref.grad)

print("aot_autograd_fw_bw_partition: ok")
