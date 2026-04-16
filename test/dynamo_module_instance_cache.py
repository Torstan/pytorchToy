import torch


compiled = {"count": 0}


class ToyModule(torch.nn.Module):
    def forward(self, x):
        return torch.relu(torch.sin(x))


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


module = ToyModule()
opt1 = torch._dynamo.optimize(backend)(module)
opt2 = torch._dynamo.optimize(backend)(module)

x = torch.randn(3, 3)
ref = module(x)

torch.testing.assert_close(opt1(x), ref)
assert compiled["count"] == 1

torch.testing.assert_close(opt2(x), ref)
assert compiled["count"] == 1, "expected the same module instance to share cache entries"

print("dynamo_module_instance_cache: ok")
