"""
Minimal torch package for pytorchToy.
Provides Tensor, factory functions, and autograd.
"""
from .tensor import Tensor, FloatTensor, zeros, ones, randn, manual_seed
from . import autograd

__all__ = [
    "Tensor", "FloatTensor", "zeros", "ones", "randn", "manual_seed",
    "autograd",
]
