import torch
import torch.fx.proxy as tracer_mod


compiled = {"count": 0}
SCALE = 2.0


def fn(x):
    y = x.sin()
    return y * SCALE + 1.0


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


original_trace = tracer_mod.Tracer.trace


def forbidden_trace(self, fn, example_inputs):
    del self, fn, example_inputs
    raise AssertionError("constant/tensor mixed bytecode should not fall back to proxy tracing")


tracer_mod.Tracer.trace = forbidden_trace

try:
    opt = torch.compile(fn, backend=backend)
    x = torch.randn(4, 4)
    ref = fn(x)
    out = opt(x)
finally:
    tracer_mod.Tracer.trace = original_trace

torch.testing.assert_close(out, ref)
assert compiled["count"] == 1

print("dynamo_constant_and_tensor_mixed: ok")
