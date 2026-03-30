import torch


torch._logging.set_logs(graph_code=True)

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return foo(a, b)


print(opt_foo2(torch.tensor([[1, 2, 3.0], [2,3,4.0], [3,4,5.0]]), torch.tensor([[1, 1, 3.0], [2,2,4.0], [4,4,6.0]])))

opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.tensor([[1, 2, 3.0], [2,3,4.0], [3,4,5.0]]), torch.tensor([[1, 1, 3.0], [2,2,4.0], [4,4,6.0]])))
