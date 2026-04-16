import torch


def demo(x):
    return torch.sin(x)


disabled = torch._dynamo.disable(demo)
assert disabled is demo

opt_disabled = torch._dynamo.optimize("eager", disable=True)(demo)
assert opt_disabled is demo

compiled = torch.compile(demo, backend="eager", disable=True)
assert compiled is demo

x = torch.randn(3, 3)
torch.testing.assert_close(compiled(x), demo(x))

print("dynamo_disable_passthrough: ok")
