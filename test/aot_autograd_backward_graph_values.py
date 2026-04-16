import torch

from torch._functorch.aot_autograd import aot_function


def eager_compiler(gm, example_inputs):
    del example_inputs

    def compiled(*args):
        return gm(*args)

    return compiled


def fn(x, y):
    return (torch.tanh(x) * y).sum()


compiled = aot_function(
    fn,
    fw_compiler=eager_compiler,
    bw_compiler=eager_compiler,
)

x = torch.tensor([1.0, -2.0])
y = torch.tensor([3.0, 4.0])
x.requires_grad = True
y.requires_grad = True

x_ref = torch.tensor([1.0, -2.0])
y_ref = torch.tensor([3.0, 4.0])
x_ref.requires_grad = True
y_ref.requires_grad = True

ref = fn(x_ref, y_ref)
ref.backward()

out = compiled(x, y)
state = compiled._last_state
assert state is not None
assert state.backward_graph_module is not None
assert state.backward_is_real is True

bw_out = state.compiled_bw(*state.backward_example_inputs)
assert isinstance(bw_out, tuple)
torch.testing.assert_close(bw_out[0], x_ref.grad)
torch.testing.assert_close(bw_out[1], y_ref.grad)

torch.testing.assert_close(out, ref)

print("aot_autograd_backward_graph_values: ok")
