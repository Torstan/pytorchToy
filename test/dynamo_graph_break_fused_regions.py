import torch


compiled = {"count": 0}


def demo(x):
    a = torch.sin(x)
    c = torch.relu(a)
    if x.item() > 0:
        b = torch.cos(c)
    else:
        b = -c
    d = torch.tanh(b)
    e = torch.relu(d)
    return e


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt = torch._dynamo.optimize(backend, nopython=False)(demo)

x_pos = torch.tensor(1.0)
torch.testing.assert_close(opt(x_pos), demo(x_pos))
assert compiled["count"] == 2, f"expected two fused compiled regions, got {compiled['count']}"

x_neg = torch.tensor(-1.0)
torch.testing.assert_close(opt(x_neg), demo(x_neg))
assert compiled["count"] == 2

print("dynamo_graph_break_fused_regions: ok")
