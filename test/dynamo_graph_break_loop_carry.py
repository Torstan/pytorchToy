import torch


compiled = {"count": 0, "runs": 0}


def demo(x):
    a = torch.sin(x)
    b = a
    for _ in range(2):
        if x.item() > 0:
            b = torch.relu(b)
        else:
            b = torch.tanh(b)
    return torch.cos(b)


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
assert compiled["count"] == 2, f"expected prefix and suffix regions to compile around the loop, got {compiled['count']}"
assert compiled["runs"] == 2, "expected prefix and suffix compiled regions on the first call"

x_neg = torch.tensor(-1.0)
torch.testing.assert_close(opt(x_neg), demo(x_neg))
assert compiled["count"] == 2, "expected cached loop-carry resume plan to be reused"
assert compiled["runs"] == 4, "expected prefix and suffix compiled regions to run again on cache hit"

print("dynamo_graph_break_loop_carry: ok")
