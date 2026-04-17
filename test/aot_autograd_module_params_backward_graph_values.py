import torch

from torch._functorch.aot_autograd import aot_module_simplified

from test_utils import eager_compiler


class ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))

    def forward(self, x):
        return (torch.tanh(x) * self.weight).sum()


module = ToyModule()
example_x = torch.tensor([1.0, 2.0])
example_x.requires_grad = True

compiled = aot_module_simplified(
    module,
    [example_x],
    fw_compiler=eager_compiler,
    bw_compiler=eager_compiler,
)

x = torch.tensor([1.0, 2.0])
x.requires_grad = True

x_ref = torch.tensor([1.0, 2.0])
x_ref.requires_grad = True
module_ref = ToyModule()

ref = module_ref(x_ref)
ref.backward()

out = compiled(x)
state = compiled._last_state
assert state is not None
assert state.backward_graph_module is not None
assert state.backward_is_real is True

bw_out = state.compiled_bw(*state.backward_example_inputs)
assert isinstance(bw_out, tuple)
assert len(bw_out) == 2

torch.testing.assert_close(out, ref)
torch.testing.assert_close(bw_out[0], module_ref.weight.grad)
torch.testing.assert_close(bw_out[1], x_ref.grad)

print("aot_autograd_module_params_backward_graph_values: ok")
