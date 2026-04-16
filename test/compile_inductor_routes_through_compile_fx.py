import torch
import importlib

compile_fx_mod = importlib.import_module("torch._inductor.compile_fx")


calls = []
original_compile_fx = compile_fx_mod.compile_fx


def wrapped_compile_fx(gm, example_inputs):
    calls.append((gm, example_inputs))
    return original_compile_fx(gm, example_inputs)


compile_fx_mod.compile_fx = wrapped_compile_fx


def fn(x, y):
    return torch.relu(torch.sin(x) + torch.cos(y))


try:
    compiled = torch.compile(fn, backend="inductor")

    x = torch.randn(4, 4)
    y = torch.randn(4, 4)

    torch.testing.assert_close(compiled(x, y), fn(x, y))
    assert len(calls) == 1, calls
finally:
    compile_fx_mod.compile_fx = original_compile_fx


print("compile_inductor_routes_through_compile_fx: ok")
