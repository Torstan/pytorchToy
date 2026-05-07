import torch
import torch.fx.proxy as tracer_mod


compiled = {"count": 0, "graphs": []}


def fn(x, scale):
    y = x + 1.0
    if scale > 0:
        y = torch.relu(y)
    return torch.sin(y)


def backend(gm, example_inputs):
    del example_inputs
    compiled["count"] += 1
    compiled["graphs"].append(gm.print_readable(print_output=False))

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


original_trace = tracer_mod.Tracer.trace


def forbidden_trace(self, fn, example_inputs):
    del self, fn, example_inputs
    raise AssertionError("static compare branch should be captured by bytecode path")


tracer_mod.Tracer.trace = forbidden_trace

try:
    opt = torch.compile(fn, backend=backend)

    x = torch.randn(4, 4)
    out_pos = opt(x, 2.0)
    ref_pos = fn(x, 2.0)

    out_neg = opt(x, -1.0)
    ref_neg = fn(x, -1.0)
finally:
    tracer_mod.Tracer.trace = original_trace

torch.testing.assert_close(out_pos, ref_pos)
torch.testing.assert_close(out_neg, ref_neg)

assert compiled["count"] == 2, compiled
assert any("relu" in graph for graph in compiled["graphs"]), compiled["graphs"]
assert any("relu" not in graph for graph in compiled["graphs"]), compiled["graphs"]

print("dynamo_bytecode_compare_branch: ok")
