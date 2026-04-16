import torch

from torch._functorch.aot_autograd import aot_module_simplified


def eager_compiler(gm, example_inputs):
    del example_inputs

    def compiled(*args):
        return gm(*args)

    return compiled


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
out.backward()

torch.testing.assert_close(out, ref)
torch.testing.assert_close(x.grad, x_ref.grad)
torch.testing.assert_close(module.weight.grad, module_ref.weight.grad)

assert compiled._last_state is not None
assert compiled._last_state.compiled_bw is not None

print("aot_autograd_module_params: ok")
