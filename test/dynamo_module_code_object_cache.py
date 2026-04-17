import torch
import torch._dynamo.eval_frame as eval_frame
from torch._dynamo.guards import GuardManager


compiled = {"count": 0}


class ScaleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

    def forward(self, x):
        return torch.sin(x) * self.scale


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


torch._dynamo.reset()

module1 = ScaleModule()
module2 = ScaleModule()

opt1 = torch._dynamo.optimize(backend)(module1)
opt2 = torch._dynamo.optimize(backend)(module2)

x = torch.randn(3, 3)
torch.testing.assert_close(opt1(x), module1(x))
torch.testing.assert_close(opt2(x), module2(x))

assert compiled["count"] == 2, f"expected separate compiled variants per module instance, got {compiled['count']}"
assert len(eval_frame._CODE_CACHE) == 1, f"expected a shared code-object cache entry, got {len(eval_frame._CODE_CACHE)}"

entry = next(iter(eval_frame._CODE_CACHE.values()))
assert len(entry.compiled_variants) == 2, (
    f"expected two guarded compiled variants in the shared cache entry, got {len(entry.compiled_variants)}"
)
assert all(isinstance(variant.guard_manager, GuardManager) for variant in entry.compiled_variants)

torch._dynamo.reset()

print("dynamo_module_code_object_cache: ok")
