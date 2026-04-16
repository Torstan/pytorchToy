import torch


compiled = {"count": 0}


def demo(x):
    return torch.relu(torch.sin(x))


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt = torch._dynamo.optimize(backend)(demo)
x = torch.randn(4, 4)

torch.testing.assert_close(opt(x), demo(x))
assert compiled["count"] == 1

torch.testing.assert_close(opt(x), demo(x))
assert compiled["count"] == 1

torch._dynamo.reset()

torch.testing.assert_close(opt(x), demo(x))
assert compiled["count"] == 2, "expected reset() to clear cached compiled results"

print("dynamo_reset_clears_cache: ok")
