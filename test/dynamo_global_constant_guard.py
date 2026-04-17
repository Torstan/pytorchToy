import torch


compiled = {"count": 0}
SCALE = 2.0


def demo(x):
    return torch.cos(x) * SCALE


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt = torch._dynamo.optimize(backend)(demo)

x = torch.randn(2, 4)
torch.testing.assert_close(opt(x), demo(x))
assert compiled["count"] == 1

SCALE = 3.0
torch.testing.assert_close(opt(x), demo(x))
assert compiled["count"] == 2, "expected recompilation after referenced global constant change"

torch.testing.assert_close(opt(x), demo(x))
assert compiled["count"] == 2

print("dynamo_global_constant_guard: ok")
