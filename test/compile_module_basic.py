import torch


class ToyModule(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(torch.sin(x) + torch.cos(y))


module = ToyModule()
compiled = torch.compile(module, backend="eager")

x = torch.randn(4, 4)
y = torch.randn(4, 4)

ref = module(x, y)
out = compiled(x, y)

torch.testing.assert_close(out, ref)

print("compile_module_basic: ok")
