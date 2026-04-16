import torch


def demo(x):
    if x.item() > 0:
        return torch.sin(x)
    return torch.cos(x)


compiled = torch.compile(demo, backend="eager", fullgraph=False)

x = torch.tensor(1.0)
torch.testing.assert_close(compiled(x), demo(x))

y = torch.tensor(-1.0)
torch.testing.assert_close(compiled(y), demo(y))

print("compile_graph_break_fallback: ok")
