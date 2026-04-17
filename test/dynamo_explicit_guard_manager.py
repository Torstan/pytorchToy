import torch
import torch._dynamo.eval_frame as eval_frame
from torch._dynamo.guards import GuardManager


compiled = {"count": 0}


def demo(x, scale):
    return torch.sin(x) * scale


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


torch._dynamo.reset()

opt = torch._dynamo.optimize(backend)(demo)

x = torch.randn(2, 3)
torch.testing.assert_close(opt(x, 2.0), demo(x, 2.0))
assert compiled["count"] == 1

assert len(eval_frame._CODE_CACHE) == 1
entry = next(iter(eval_frame._CODE_CACHE.values()))
assert len(entry.compiled_variants) == 1

variant = entry.compiled_variants[0]
assert isinstance(variant.guard_manager, GuardManager)

lines = variant.guard_manager.describe()
assert any(line == "arg0: tensor shape=(2, 3) dtype=float32 requires_grad=False contiguous=True" for line in lines), lines
assert any(line == "arg1: float value=2.0" for line in lines), lines

torch._dynamo.reset()

print("dynamo_explicit_guard_manager: ok")
