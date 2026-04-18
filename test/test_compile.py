import torch

# 1. 设置日志
torch._logging.set_logs(graph_code=True)

# 2. 您的代码...
class A(torch.nn.Module):
    def forward(self, x):
        return torch.exp(2 * x)

def f(x, mod):
    y = mod(x)
    z = torch.log(y)
    return z

a = torch.compile(f, backend="inductor")

# 3. 运行以触发编译
x = torch.randn(3, 3)
mod = A()
result = a(x, mod)  # 这里会打印图代码
