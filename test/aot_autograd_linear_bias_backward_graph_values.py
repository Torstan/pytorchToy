import torch

from torch._functorch.aot_autograd import aot_function


def eager_compiler(gm, example_inputs):
    del example_inputs

    def compiled(*args):
        return gm(*args)

    return compiled


def fn(x, w, b):
    return torch.relu(x.mm(w) + b).sum()


compiled = aot_function(
    fn,
    fw_compiler=eager_compiler,
    bw_compiler=eager_compiler,
)

x = torch.randn(4, 3)
w = torch.randn(3, 5)
b = torch.randn(5)
x.requires_grad = True
w.requires_grad = True
b.requires_grad = True

x_ref = x.clone()
w_ref = w.clone()
b_ref = b.clone()
x_ref.requires_grad = True
w_ref.requires_grad = True
b_ref.requires_grad = True

ref = fn(x_ref, w_ref, b_ref)
ref.backward()

out = compiled(x, w, b)
state = compiled._last_state
assert state is not None
assert state.backward_graph_module is not None
assert state.backward_is_real is True

bw_out = state.compiled_bw(*state.backward_example_inputs)
assert isinstance(bw_out, tuple)
assert len(bw_out) == 3

torch.testing.assert_close(out, ref)
torch.testing.assert_close(bw_out[0], x_ref.grad)
torch.testing.assert_close(bw_out[1], w_ref.grad)
torch.testing.assert_close(bw_out[2], b_ref.grad)

print("aot_autograd_linear_bias_backward_graph_values: ok")
