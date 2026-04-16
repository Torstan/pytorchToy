import torch


module = torch.nn.LayerNorm(5)
compiled = torch.compile(module, backend="inductor")

x = torch.randn(4, 5)
x.requires_grad = True

x_ref = x.clone()
x_ref.requires_grad = True

module_ref = torch.nn.LayerNorm(5)
module_ref.weight = torch.nn.Parameter(module.weight.clone())
module_ref.bias = torch.nn.Parameter(module.bias.clone())

out = compiled(x).sum()
out.backward()

ref = module_ref(x_ref).sum()
ref.backward()

torch.testing.assert_close(out, ref)
torch.testing.assert_close(x.grad, x_ref.grad)
torch.testing.assert_close(module.weight.grad, module_ref.weight.grad)
torch.testing.assert_close(module.bias.grad, module_ref.bias.grad)

print("compile_training_layer_norm_module_inductor: ok")
