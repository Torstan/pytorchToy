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
