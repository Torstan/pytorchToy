import torch


compiled = {"count": 0}


class ScaleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

    def forward(self, x):
        return torch.sin(x) * self.scale


def fn(x, mod):
    return mod(x)


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt = torch.compile(fn, backend=backend)

x = torch.randn(3, 3)
mod = ScaleModule()

torch.testing.assert_close(opt(x, mod), fn(x, mod))
assert compiled["count"] == 1

mod.scale = 3.0
torch.testing.assert_close(opt(x, mod), fn(x, mod))
assert compiled["count"] == 2, "expected recompilation after specialized module arg state change"

torch.testing.assert_close(opt(x, mod), fn(x, mod))
assert compiled["count"] == 2

print("compile_module_arg_guard_recompile: ok")
