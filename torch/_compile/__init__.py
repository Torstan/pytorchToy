"""
torch.compile API -- pytorchToy 的编译前端。

当前这一层只保留公共 API 形状，把调用转发到
torch._dynamo.optimize(...)。
"""


def compile(model=None, *, fullgraph=False, dynamic=None, backend="default",
            mode=None, options=None, disable=False):
    """
    torch.compile -- 编译优化函数或模块

    对应 PyTorch torch.compile()

    支持三种用法:
      1. opt_fn = torch.compile(fn)         # 直接包装
      2. @torch.compile                     # 装饰器
      3. @torch.compile(backend="eager")    # 带参数的装饰器

    Args:
        model: 要编译的函数或 Module (None 时返回装饰器)
        fullgraph: 是否要求整个函数为单一图
        dynamic: 动态形状 (简化实现中未使用)
        backend: 后端名称或 callable
        mode: 编译模式 (简化实现中未使用)
        options: 后端选项 (简化实现中未使用)
        disable: 是否禁用编译 (True 时返回原函数)
    """
    import torch

    decorator = torch._dynamo.optimize(
        backend=backend,
        nopython=fullgraph,
        dynamic=dynamic,
        disable=disable,
    )

    if model is not None and callable(model):
        return decorator(model)

    return decorator
