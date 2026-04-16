import torch

from torch._functorch.aot_autograd import aot_function


def eager_compiler(gm, example_inputs):
    del gm, example_inputs
    raise AssertionError("compiler should not run for mutation case")


def fn(x):
    x += 1.0
    return torch.tanh(x)


compiled = aot_function(
    fn,
    fw_compiler=eager_compiler,
    bw_compiler=eager_compiler,
)

try:
    compiled(torch.tensor([1.0, 2.0]))
except NotImplementedError as exc:
    assert "mutation" in str(exc).lower()
    print("aot_autograd_mutation_rejected: ok")
else:
    raise AssertionError("expected input mutation to be rejected")
