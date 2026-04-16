import torch


calls = []
original_optimize = torch._dynamo.optimize


def spy_optimize(*args, **kwargs):
    calls.append((args, kwargs))
    return original_optimize(*args, **kwargs)


torch._dynamo.optimize = spy_optimize

try:
    def demo(x, y):
        return torch.sin(x) + torch.cos(y)

    compiled = torch.compile(demo, backend="eager")
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    out = compiled(x, y)
    ref = demo(x, y)
    torch.testing.assert_close(out, ref)
finally:
    torch._dynamo.optimize = original_optimize

assert len(calls) == 1, f"expected torch.compile to call torch._dynamo.optimize once, got {len(calls)}"
assert calls[0][1]["backend"] == "eager"

print("compile_routes_through_dynamo: ok")
