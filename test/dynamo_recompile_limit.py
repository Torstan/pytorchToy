import torch


compiled = {"count": 0}
prior_limit = torch._dynamo.config.recompile_limit


def demo(x):
    return torch.tanh(torch.sin(x) + 1.0)


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


try:
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 1
    opt = torch._dynamo.optimize(backend)(demo)

    x1 = torch.randn(2, 2)
    x2 = torch.randn(3, 3)

    torch.testing.assert_close(opt(x1), demo(x1))
    assert compiled["count"] == 1

    torch.testing.assert_close(opt(x2), demo(x2))
    assert compiled["count"] == 1, "expected eager fallback after hitting recompile limit"
finally:
    torch._dynamo.config.recompile_limit = prior_limit
    torch._dynamo.reset()

print("dynamo_recompile_limit: ok")
