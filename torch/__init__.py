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


def sin(input_tensor):
    """torch.sin -- 逐元素正弦函数"""
    if hasattr(input_tensor, "sin"):
        return input_tensor.sin()
    raise TypeError(f"torch.sin expected Tensor-like input, got {type(input_tensor)}")


def cos(input_tensor):
    """torch.cos -- 逐元素余弦函数"""
    if hasattr(input_tensor, "cos"):
        return input_tensor.cos()
    raise TypeError(f"torch.cos expected Tensor-like input, got {type(input_tensor)}")
