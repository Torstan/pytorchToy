import torch


compiled = {"count": 0}
SCALE = 2.0


class ScaleModule(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) * SCALE


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


module = ScaleModule()
opt = torch._dynamo.optimize(backend)(module)

x = torch.randn(2, 4)
torch.testing.assert_close(opt(x), module(x))
assert compiled["count"] == 1

SCALE = 3.0
torch.testing.assert_close(opt(x), module(x))
assert compiled["count"] == 2, "expected recompilation after module forward global constant change"

torch.testing.assert_close(opt(x), module(x))
assert compiled["count"] == 2

print("dynamo_module_forward_global_guard: ok")
