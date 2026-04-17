import torch


compiled = {"count": 0}


class ScaleBox:
    def __init__(self):
        self.scale = 2.0

    def forward(self, x):
        return torch.sin(x) * self.scale


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


box = ScaleBox()
opt = torch.compile(box.forward, backend=backend)

x = torch.randn(2, 3)
torch.testing.assert_close(opt(x), box.forward(x))
assert compiled["count"] == 1

box.scale = 3.0
torch.testing.assert_close(opt(x), box.forward(x))
assert compiled["count"] == 2, "expected recompilation after bound-method self state change"

torch.testing.assert_close(opt(x), box.forward(x))
assert compiled["count"] == 2

print("dynamo_bound_method_guard: ok")
