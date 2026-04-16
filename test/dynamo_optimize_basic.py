import torch


compiled = {"count": 0, "gm": None, "inputs": None}


def eager_fn(x, y):
    return torch.relu(torch.sin(x) + torch.cos(y))


def backend(gm, example_inputs):
    compiled["count"] += 1
    compiled["gm"] = gm
    compiled["inputs"] = list(example_inputs)

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt_fn = torch._dynamo.optimize(backend)(eager_fn)

x = torch.randn(4, 4)
y = torch.randn(4, 4)
ref = eager_fn(x, y)
out = opt_fn(x, y)

torch.testing.assert_close(out, ref)
assert compiled["count"] == 1
assert compiled["gm"] is not None
assert len(compiled["inputs"]) == 2
assert compiled["inputs"][0].shape == (4, 4)

readable = compiled["gm"].print_readable(print_output=False)
assert "torch.sin" in readable
assert "torch.cos" in readable
assert "relu" in readable

second = opt_fn(x, y)
torch.testing.assert_close(second, ref)
assert compiled["count"] == 1, "expected cache hit on same signature"

print("dynamo_optimize_basic: ok")
