import torch


compiled = {"count": 0}


def make_callable():
    scale = 2.0

    class ScaleCallable:
        def __init__(self):
            self.bias = 1.0

        def __call__(self, x):
            return torch.sin(x) * scale + self.bias

    def set_scale(value):
        nonlocal scale
        scale = value

    return ScaleCallable(), set_scale


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


callable_obj, set_scale = make_callable()
opt = torch._dynamo.optimize(backend)(callable_obj)

x = torch.randn(2, 4)
torch.testing.assert_close(opt(x), callable_obj(x))
assert compiled["count"] == 1

set_scale(3.0)
torch.testing.assert_close(opt(x), callable_obj(x))
assert compiled["count"] == 2, "expected recompilation after callable __call__ closure change"

callable_obj.bias = 2.0
torch.testing.assert_close(opt(x), callable_obj(x))
assert compiled["count"] == 3, "expected recompilation after callable object attribute change"

torch.testing.assert_close(opt(x), callable_obj(x))
assert compiled["count"] == 3

print("dynamo_callable_object_guards: ok")
