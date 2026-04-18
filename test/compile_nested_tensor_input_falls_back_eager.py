import torch


compiled = {"count": 0}


def fn(inputs):
    x = inputs["x"]
    y = inputs["y"]
    x = x.cos().cos()
    if x.mean() > 0.5:
        x = x / 1.1
    return x * y


def backend(gm, example_inputs):
    del gm, example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        raise AssertionError("nested tensor container input should fall back to eager")

    return compiled_fn


opt = torch.compile(fn, backend=backend)

inputs = {
    "x": torch.randn(10, requires_grad=True),
    "y": torch.randn(10, requires_grad=True),
}

torch.testing.assert_close(opt(inputs), fn(inputs))
assert compiled["count"] == 0, compiled

print("compile_nested_tensor_input_falls_back_eager: ok")
