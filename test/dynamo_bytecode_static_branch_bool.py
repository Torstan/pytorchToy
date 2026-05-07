import torch
import torch.fx.proxy as tracer_mod


compiled = {"count": 0, "graphs": []}


def fn(x, flag):
    y = torch.sin(x)
    if flag:
        y = torch.relu(y)
    else:
        y = torch.tanh(y)
    return torch.cos(y)


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
    raise AssertionError("static bool branch should be captured by bytecode path")


tracer_mod.Tracer.trace = forbidden_trace

try:
    opt = torch.compile(fn, backend=backend)

    x = torch.randn(4, 4)
    out_true = opt(x, True)
    ref_true = fn(x, True)

    out_false = opt(x, False)
    ref_false = fn(x, False)
finally:
    tracer_mod.Tracer.trace = original_trace

torch.testing.assert_close(out_true, ref_true)
torch.testing.assert_close(out_false, ref_false)

assert compiled["count"] == 2, compiled
assert any("relu" in graph for graph in compiled["graphs"]), compiled["graphs"]
assert any("tanh" in graph for graph in compiled["graphs"]), compiled["graphs"]

print("dynamo_bytecode_static_branch_bool: ok")
