import torch

from torch._functorch.aot_autograd import aot_function


counts = {"bw_compile": 0, "bw_run": 0}


def eager_compiler(gm, example_inputs):
    del example_inputs

    def compiled(*args):
        return gm(*args)

    return compiled


def counting_bw_compiler(gm, example_inputs):
    del example_inputs
    counts["bw_compile"] += 1

    def compiled(*args):
        counts["bw_run"] += 1
        return gm(*args)

    return compiled


def fn(x, y):
    return (torch.tanh(x) * y).sum()


compiled = aot_function(
    fn,
    fw_compiler=eager_compiler,
    bw_compiler=counting_bw_compiler,
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
assert counts == {"bw_compile": 1, "bw_run": 0}, counts

out.backward()

assert counts == {"bw_compile": 1, "bw_run": 1}, counts
torch.testing.assert_close(x.grad, x_ref.grad)
torch.testing.assert_close(y.grad, y_ref.grad)

print("aot_autograd_runtime_compiled_backward: ok")
