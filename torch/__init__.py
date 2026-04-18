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
import torch._dynamo
import torch._logging
import torch.testing


def set_num_threads(num):
    """设置线程数 (pytorchToy 无多线程运行时，仅做兼容)"""
    pass


def get_num_threads():
    """获取线程数"""
    return 1


def relu(input_tensor):
    """torch.relu -- 逐元素 ReLU"""
    if hasattr(input_tensor, "relu"):
        return input_tensor.relu()
    raise TypeError(f"torch.relu expected Tensor-like input, got {type(input_tensor)}")


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


def tanh(input_tensor):
    """torch.tanh -- 逐元素双曲正切函数"""
    if hasattr(input_tensor, "tanh"):
        return input_tensor.tanh()
    raise TypeError(f"torch.tanh expected Tensor-like input, got {type(input_tensor)}")


def exp(input_tensor):
    """torch.exp -- 逐元素指数函数"""
    if hasattr(input_tensor, "exp"):
        return input_tensor.exp()
    raise TypeError(f"torch.exp expected Tensor-like input, got {type(input_tensor)}")


def log(input_tensor):
    """torch.log -- 逐元素对数函数"""
    if hasattr(input_tensor, "log"):
        return input_tensor.log()
    raise TypeError(f"torch.log expected Tensor-like input, got {type(input_tensor)}")
