import torch


compiled = {"count": 0, "runs": 0}


def demo(x):
    a = torch.sin(x)
    if x.item() > 0:
        return torch.relu(a)
    return torch.tanh(a)


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        compiled["runs"] += 1
        return gm(*args)

    return compiled_fn


opt = torch._dynamo.optimize(backend, nopython=False)(demo)

x_pos = torch.tensor(1.0)
torch.testing.assert_close(opt(x_pos), demo(x_pos))
assert compiled["count"] == 2, f"expected prefix and fallthrough return regions to compile, got {compiled['count']}"
assert compiled["runs"] == 1, "expected only the prefix region to run on the early-return path"

x_neg = torch.tensor(-1.0)
torch.testing.assert_close(opt(x_neg), demo(x_neg))
assert compiled["count"] == 2, "expected cached early-return resume plan to be reused"
assert compiled["runs"] == 3, "expected prefix and fallthrough return regions to run on the non-early-return path"

print("dynamo_graph_break_early_return: ok")
