import torch


compiled = {"count": 0}


class ScaleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

    def forward(self, x):
        return torch.sin(x) * self.scale


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


module = ScaleModule()
opt = torch._dynamo.optimize(backend)(module)

x = torch.randn(3, 3)
torch.testing.assert_close(opt(x), module(x))
assert compiled["count"] == 1

module.scale = 3.0
torch.testing.assert_close(opt(x), module(x))
assert compiled["count"] == 2, "expected recompilation after module scalar attribute change"

torch.testing.assert_close(opt(x), module(x))
assert compiled["count"] == 2

print("dynamo_module_attr_guard: ok")
