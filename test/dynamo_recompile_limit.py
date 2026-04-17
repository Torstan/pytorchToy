import torch
import torch._dynamo.eval_frame as eval_frame
from torch._dynamo.guards import GuardManager


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

    assert len(eval_frame._CODE_CACHE) == 1
    entry = next(iter(eval_frame._CODE_CACHE.values()))
    assert len(entry.compiled_variants) == 1
    assert len(entry.eager_fallback_variants) == 1
    assert isinstance(entry.compiled_variants[0].guard_manager, GuardManager)
    assert isinstance(entry.eager_fallback_variants[0], GuardManager)
finally:
    torch._dynamo.config.recompile_limit = prior_limit
    torch._dynamo.reset()

print("dynamo_recompile_limit: ok")
