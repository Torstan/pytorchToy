"""
最小 refs 实现。
"""

import torch._prims as prims


def relu(x):
    return prims.relu(x)


def addmm(bias, lhs, rhs):
    return prims.add(prims.mm(lhs, rhs), bias)


def linear(x, weight, bias=None):
    out = prims.mm(x, weight.t())
    if bias is None:
        return out
    return prims.add(out, bias)
