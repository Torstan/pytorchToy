"""
Python Tensor 包装类
包装 _C.Tensor，提供 Python 风格的接口
"""

import sys
import os
import math
import random

# 导入 C++ 扩展模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _C


class Tensor:
    """Python 侧的 Tensor 包装类，包装 _C.Tensor"""

    def __init__(self, c_tensor=None):
        if c_tensor is None:
            self._c = _C.Tensor([0])
        elif isinstance(c_tensor, _C.Tensor):
            self._c = c_tensor
        elif isinstance(c_tensor, Tensor):
            self._c = c_tensor._c
        else:
            raise TypeError(f"Tensor: unsupported init type {type(c_tensor)}")

    # ---- 基本属性 ----

    def dim(self):
        return self._c.dim()

    def numel(self):
        return self._c.numel()

    def size(self, dim=None):
        s = list(self._c.sizes())
        if dim is not None:
            return s[dim]
        return s

    def item(self):
        if self.dim() == 0:
            return self._c.item()
        if self.numel() == 1:
            # 对于 1 元素张量，也支持 item()
            c = self._c
            while c.dim() > 0:
                c = c[0]
            return c.item()
        raise RuntimeError("item() only for single-element tensors")

    def data_ptr(self):
        return self._c.data_ptr_id()

    def is_contiguous(self):
        return self._c.is_contiguous()

    def clone(self):
        """深拷贝"""
        sizes = list(self._c.sizes())
        result = _C.empty(sizes)
        n = self.numel()
        for i in range(n):
            result.flat_set(i, self._c.flat_get(i))
        return Tensor(result)

    @property
    def data(self):
        """兼容 Variable 接口：Tensor.data 返回自身"""
        return self

    @property
    def _version(self):
        """版本计数器，用于 autograd in-place 检测"""
        return self._c._version()

    # ---- 索引 ----

    def __getitem__(self, idx):
        if isinstance(idx, int):
            result = Tensor(self._c[idx])
            # 0-dim 结果返回 Python float（兼容旧版 PyTorch 行为）
            if result.dim() == 0:
                return result.item()
            return result
        elif isinstance(idx, tuple):
            result = self._c
            for i in idx:
                result = result[i]
            t = Tensor(result)
            if t.dim() == 0:
                return t.item()
            return t
        raise TypeError(f"unsupported index type: {type(idx)}")

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            if isinstance(value, (int, float)):
                self._c[idx] = float(value)
                self._c._bump_version()
            else:
                raise TypeError("only scalar assignment supported")
        else:
            raise TypeError(f"unsupported index type: {type(idx)}")

    def __len__(self):
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.size(0)

    # ---- 算术运算 ----

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._c.add(other._c))
        elif isinstance(other, (int, float)):
            return self._apply_scalar(float(other), lambda a, b: a + b)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, Tensor):
            result = self._c.add(other._c)
            self._c = result
        elif isinstance(other, (int, float)):
            self._inplace_scalar(float(other), lambda a, b: a + b)
        self._c._bump_version()
        return self

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self._apply_binary(other, lambda a, b: a - b)
        elif isinstance(other, (int, float)):
            return self._apply_scalar(float(other), lambda a, b: a - b)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self._apply_scalar(float(other), lambda a, b: b - a)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self._c.mul(other._c))
        elif isinstance(other, (int, float)):
            return self._apply_scalar(float(other), lambda a, b: a * b)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self._apply_scalar(1.0 / float(other), lambda a, b: a * b)
        elif isinstance(other, Tensor):
            return self._apply_binary(other, lambda a, b: a / b if b != 0 else 0.0)
        return NotImplemented

    def __neg__(self):
        return self._apply_scalar(-1.0, lambda a, b: a * b)

    # ---- 矩阵运算 ----

    def mm(self, other):
        """矩阵乘法"""
        return Tensor(self._c.matmul(other._c))

    def t(self):
        """转置 (2D)"""
        if self.dim() != 2:
            raise RuntimeError("t() expects a 2D tensor")
        return Tensor(self._c.transpose(0, 1))

    # ---- 变形 ----

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        return Tensor(self._c.reshape(shape))

    def reshape(self, *shape):
        return self.view(*shape)

    def unsqueeze(self, dim):
        """在指定维度插入大小为 1 的维度"""
        sizes = list(self._c.sizes())
        if dim < 0:
            dim = len(sizes) + 1 + dim
        new_sizes = sizes[:dim] + [1] + sizes[dim:]
        return self.view(new_sizes)

    def expand(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = list(sizes[0])
        else:
            sizes = list(sizes)
        return Tensor(self._c.expand(sizes))

    def expand_as(self, other):
        return self.expand(other.size())

    def contiguous(self):
        return Tensor(self._c.contiguous())

    # ---- 比较 ----

    def gt(self, value):
        """逐元素 > value"""
        return self._apply_scalar(float(value), lambda a, b: 1.0 if a > b else 0.0)

    def __ne__(self, other):
        if isinstance(other, Tensor):
            return self._apply_binary(other, lambda a, b: 1.0 if a != b else 0.0)
        return self._apply_scalar(float(other), lambda a, b: 1.0 if a != b else 0.0)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self._apply_binary(other, lambda a, b: 1.0 if a == b else 0.0)
        return self._apply_scalar(float(other), lambda a, b: 1.0 if a == b else 0.0)

    # ---- 归约 ----

    def sum(self, dim=None):
        if dim is None:
            # 全局 sum → 标量张量
            val = _C.sum(self._c)
            result = _C.Tensor([1], val)
            # 将 [1] reshape 为 []: 不支持 0-dim, 返回 1-dim
            return Tensor(result)
        else:
            # 沿维度 sum
            return self._sum_dim(dim)

    def any(self):
        """是否有非零元素"""
        n = self.numel()
        for i in range(n):
            if self._c.flat_get(i) != 0.0:
                return True
        return False

    # ---- 其他 ----

    def clamp(self, min=None, max=None):
        sizes = list(self._c.sizes())
        result = _C.empty(sizes)
        n = self.numel()
        for i in range(n):
            v = self._c.flat_get(i)
            if min is not None and v < min:
                v = float(min)
            if max is not None and v > max:
                v = float(max)
            result.flat_set(i, v)
        return Tensor(result)

    def float(self):
        """返回自身（已经是 float 类型）"""
        return self

    def __float__(self):
        return float(self.item())

    def __bool__(self):
        return bool(self.item())

    def __repr__(self):
        return f"Tensor({self._c.__repr__()})"

    # ---- 内部辅助方法 ----

    def _read_elem(self, flat_idx):
        """按逻辑索引读取元素（stride 感知，委托给 C++）"""
        return self._c.flat_get(flat_idx)

    def _apply_scalar(self, scalar, op):
        """对每个元素应用标量运算"""
        sizes = list(self._c.sizes())
        result = _C.empty(sizes)
        n = self.numel()
        for i in range(n):
            result.flat_set(i, op(self._c.flat_get(i), scalar))
        return Tensor(result)

    def _inplace_scalar(self, scalar, op):
        n = self.numel()
        for i in range(n):
            self._c.flat_set(i, op(self._c.flat_get(i), scalar))
        self._c._bump_version()

    def _apply_binary(self, other, op):
        """逐元素二元运算"""
        sizes = list(self._c.sizes())
        result = _C.empty(sizes)
        n = self.numel()
        for i in range(n):
            result.flat_set(i, op(self._c.flat_get(i), other._c.flat_get(i)))
        return Tensor(result)

    def _sum_dim(self, dim):
        """沿指定维度求和"""
        sizes = list(self._c.sizes())
        if dim < 0:
            dim = len(sizes) + dim

        # 结果形状：去掉 dim 维度
        new_sizes = sizes[:dim] + sizes[dim + 1:]
        if not new_sizes:
            new_sizes = [1]

        result = _C.Tensor(new_sizes, 0.0)

        # 先确保连续
        c = self._c if self._c.is_contiguous() else self._c.contiguous()

        n_outer = 1
        for i in range(dim):
            n_outer *= sizes[i]
        n_dim = sizes[dim]
        n_inner = 1
        for i in range(dim + 1, len(sizes)):
            n_inner *= sizes[i]

        for outer in range(n_outer):
            for inner in range(n_inner):
                s = 0.0
                for d in range(n_dim):
                    idx = outer * n_dim * n_inner + d * n_inner + inner
                    s += c.flat_get(idx)
                result.flat_set(outer * n_inner + inner, s)

        return Tensor(result)


# ============================================================
# 工厂函数
# ============================================================

def FloatTensor(data):
    """从嵌套 list 创建 Tensor"""
    if isinstance(data, (list, tuple)):
        c = _C.Tensor.from_data(list(data))
        return Tensor(c)
    raise TypeError(f"FloatTensor: unsupported type {type(data)}")


def zeros(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)
    return Tensor(_C.Tensor(shape, 0.0))


def ones(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)
    return Tensor(_C.ones(shape))


_rng_seed = None


def manual_seed(seed):
    global _rng_seed
    _rng_seed = seed
    random.seed(seed)


def randn(*shape):
    """生成标准正态分布随机张量（Box-Muller 变换）"""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)

    n = 1
    for s in shape:
        n *= s

    result = _C.empty(shape)
    # Box-Muller 变换
    for i in range(0, n, 2):
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        result.flat_set(i, z0)
        if i + 1 < n:
            result.flat_set(i + 1, z1)

    return Tensor(result)
