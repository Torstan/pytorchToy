import torch


compiled = {"count": 0}


def demo(x):
    return torch.relu(torch.sin(x) + 1.0)


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt = torch._dynamo.optimize(backend)(demo)

x1 = torch.randn(2, 3)
x2 = torch.randn(4, 3)

torch.testing.assert_close(opt(x1), demo(x1))
assert compiled["count"] == 1

torch.testing.assert_close(opt(x2), demo(x2))
assert compiled["count"] == 2, "expected recompilation for a new input shape"

torch.testing.assert_close(opt(x2), demo(x2))
assert compiled["count"] == 2

print("dynamo_recompile_on_shape_change: ok")
