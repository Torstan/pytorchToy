import torch

from torch._functorch.aot_autograd import aot_function


seen = {"fw": 0, "bw": 0, "inf": 0}


def make_compiler(name):
    def compiler(gm, example_inputs):
        del example_inputs
        seen[name] += 1

        def compiled(*args):
            return gm(*args)

        return compiled

    return compiler


def fn(x, y):
    return torch.relu(x + y)


compiled = aot_function(
    fn,
    fw_compiler=make_compiler("fw"),
    bw_compiler=make_compiler("bw"),
    inference_compiler=make_compiler("inf"),
)

x = torch.randn(4, 4)
y = torch.randn(4, 4)

ref = fn(x, y)
out = compiled(x, y)
torch.testing.assert_close(out, ref)

out_again = compiled(x, y)
torch.testing.assert_close(out_again, ref)

assert seen == {"fw": 0, "bw": 0, "inf": 1}, seen

print("aot_autograd_inference_basic: ok")
