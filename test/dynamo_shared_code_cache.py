import torch


compiled = {"count": 0}


def demo(x, y):
    return torch.sin(x) + torch.cos(y)


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt1 = torch._dynamo.optimize(backend)(demo)
opt2 = torch._dynamo.optimize(backend)(demo)

x = torch.randn(3, 3)
y = torch.randn(3, 3)
ref = demo(x, y)

torch.testing.assert_close(opt1(x, y), ref)
assert compiled["count"] == 1

torch.testing.assert_close(opt2(x, y), ref)
assert compiled["count"] == 1, "expected wrappers over the same code object to share cache"

print("dynamo_shared_code_cache: ok")
