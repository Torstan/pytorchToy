import torch


compiled = {"count": 0}


def fn(x, *, scale=1.0):
    return torch.sin(x) * scale


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt = torch.compile(fn, backend=backend)

x = torch.randn(2, 3)
torch.testing.assert_close(opt(x, scale=2.0), fn(x, scale=2.0))
assert compiled["count"] == 1

torch.testing.assert_close(opt(x, scale=2.0), fn(x, scale=2.0))
assert compiled["count"] == 1, "expected kwargs call to hit cache"

print("compile_kwargs_basic: ok")
