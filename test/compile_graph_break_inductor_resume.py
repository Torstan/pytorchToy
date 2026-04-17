import importlib

import torch


compile_fx_mod = importlib.import_module("torch._inductor.compile_fx")


calls = []
original_compile_fx = compile_fx_mod.compile_fx


def wrapped_compile_fx(gm, example_inputs):
    calls.append((gm, example_inputs))
    return original_compile_fx(gm, example_inputs)


compile_fx_mod.compile_fx = wrapped_compile_fx


def fn(x):
    a = torch.sin(x)
    c = torch.relu(a)
    if x.item() > 0:
        b = torch.cos(c)
    else:
        b = -c
    d = torch.tanh(b)
    e = torch.relu(d)
    return e


try:
    compiled = torch.compile(fn, backend="inductor", fullgraph=False)

    x_pos = torch.tensor(1.0)
    x_neg = torch.tensor(-1.0)

    torch.testing.assert_close(compiled(x_pos), fn(x_pos))
    assert len(calls) == 2, f"expected two compile_fx calls for two fused regions, got {len(calls)}"

    torch.testing.assert_close(compiled(x_neg), fn(x_neg))
    assert len(calls) == 2
finally:
    compile_fx_mod.compile_fx = original_compile_fx


print("compile_graph_break_inductor_resume: ok")
