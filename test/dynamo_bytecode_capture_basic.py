import torch
import torch.fx.proxy as tracer_mod


compiled = {"count": 0, "gm": None}


def fn(x):
    y = torch.sin(x)
    z = y + 1.0
    return torch.relu(z)


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1
    compiled["gm"] = gm

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


original_trace = tracer_mod.Tracer.trace


def forbidden_trace(self, fn, example_inputs):
    del self, fn, example_inputs
    raise AssertionError("simple bytecode capture should not fall back to proxy tracing")


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
assert compiled["gm"] is not None

readable = compiled["gm"].print_readable(print_output=False)
assert "torch.sin" in readable
assert "relu" in readable

print("dynamo_bytecode_capture_basic: ok")
