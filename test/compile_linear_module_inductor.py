import torch


module = torch.nn.Linear(3, 5)
compiled = torch.compile(module, backend="inductor")

x = torch.randn(4, 3)

torch.testing.assert_close(compiled(x), module(x))

print("compile_linear_module_inductor: ok")
