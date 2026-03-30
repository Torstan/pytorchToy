"""
pytorchToy 的 Python 包入口
模拟 PyTorch 的 import torch 接口
"""

from torch.tensor import (
    Tensor, FloatTensor, zeros, ones, randn, manual_seed,
    tensor, randint, argmax,
    float32, long,
)
import torch.autograd
from torch.autograd import Variable
import torch.nn
import torch.optim
from torch.autograd_engine import no_grad
from torch._compile import compile
import torch._logging


def sin(x):
    """torch.sin -- 逐元素正弦函数"""
    from torch._compile.tracer import is_tracing, get_current_tracer, Proxy, create_proxy_for_function
    if isinstance(x, Proxy):
        return create_proxy_for_function(get_current_tracer(), sin, (x,))
    return x.sin()


def cos(x):
    """torch.cos -- 逐元素余弦函数"""
    from torch._compile.tracer import is_tracing, get_current_tracer, Proxy, create_proxy_for_function
    if isinstance(x, Proxy):
        return create_proxy_for_function(get_current_tracer(), cos, (x,))
    return x.cos()
