import torch


compiled = {"count": 0, "runs": 0}


def demo(x):
    a = torch.sin(x)
    if x.item() > 0:
        b = torch.cos(a)
    else:
        return torch.tanh(a)
    return torch.relu(b)


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
assert compiled["count"] == 2, f"expected prefix and suffix regions to compile, got {compiled['count']}"
assert compiled["runs"] == 2, "expected prefix and suffix compiled regions on the fallthrough path"

x_neg = torch.tensor(-1.0)
torch.testing.assert_close(opt(x_neg), demo(x_neg))
assert compiled["count"] == 2, "expected cached mixed early-return resume plan to be reused"
assert compiled["runs"] == 3, "expected only the prefix region to run on the early-return path"

print("dynamo_graph_break_mixed_early_return: ok")
