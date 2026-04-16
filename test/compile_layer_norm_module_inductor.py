import torch


module = torch.nn.LayerNorm(5)
compiled = torch.compile(module, backend="inductor")

x = torch.randn(4, 5)

torch.testing.assert_close(compiled(x), module(x))

print("compile_layer_norm_module_inductor: ok")
