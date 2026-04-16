import importlib

import torch


compile_fx_mod = importlib.import_module("torch._inductor.compile_fx")


counts = {
    "compile": 0,
    "run": 0,
}

original_compile_graph_module = compile_fx_mod.compile_graph_module


def wrapped_compile_graph_module(gm, example_inputs, **kwargs):
    compiled = original_compile_graph_module(gm, example_inputs, **kwargs)
    counts["compile"] += 1

    class WrappedKernel:
        def run(self, args):
            counts["run"] += 1
            return compiled.run(args)

    return WrappedKernel()


compile_fx_mod.compile_graph_module = wrapped_compile_graph_module


def fn(x, y):
    return torch.tanh(x) * y


try:
    compiled = torch.compile(fn, backend="inductor")

    x = torch.tensor([1.0, -2.0])
    y = torch.tensor([3.0, 4.0])
    x.requires_grad = True

    x_ref = torch.tensor([1.0, -2.0])
    y_ref = torch.tensor([3.0, 4.0])
    x_ref.requires_grad = True

    ref = fn(x_ref, y_ref)
    ref.backward()

    out = compiled(x, y)
    assert counts == {"compile": 2, "run": 2}, counts

    out.backward()

    assert counts == {"compile": 2, "run": 3}, counts
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(x.grad, x_ref.grad)
finally:
    compile_fx_mod.compile_graph_module = original_compile_graph_module


print("compile_training_inductor_pointwise_compiles_fw_bw: ok")
