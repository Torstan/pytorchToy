"""
Tensor class wrapping _C.Tensor with full operator support.

Provides the operations needed by autograd Functions:
  mm, t, unsqueeze, expand_as, clamp, gt, sum(dim), view, size,
  clone, scalar arithmetic, in-place add, ne, any, float, etc.
"""
import sys
import os

# Ensure the parent directory (pytorchToy/) is on the path so _C can be imported
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import _C
import math
import random

# Global RNG state
_rng = random.Random()


def manual_seed(seed):
    _rng.seed(seed)


class Tensor:
    """Python Tensor wrapping _C.Tensor, with full operator suite."""

    def __init__(self, data=None, _c=None):
        """
        Tensor(list)        → from nested list
        Tensor(_c=c_tensor) → wrap existing _C.Tensor
        """
        if _c is not None:
            self._c = _c
        elif data is not None:
            if isinstance(data, _C.Tensor):
                self._c = data
            elif isinstance(data, Tensor):
                self._c = data._c
            elif isinstance(data, (list, tuple)):
                self._c = _C.tensor(data)
            else:
                raise TypeError(f"Cannot create Tensor from {type(data)}")
        else:
            raise TypeError("Tensor() requires data or _c argument")
        self._version = 0

    # --- shape helpers ---
    def dim(self):
        return self._c.dim()

    def size(self, dim=None):
        s = self._c.sizes()
        if dim is not None:
            return s[dim]
        return tuple(s)

    def numel(self):
        return self._c.numel()

    def is_contiguous(self):
        return self._c.is_contiguous()

    # --- element access ---
    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            # multi-dim: t[i,j,...]
            r = self._c
            for i in idx:
                r = r[i]
        else:
            r = self._c[idx]
        # If 0-dim, return python float
        if isinstance(r, _C.Tensor) and r.dim() == 0:
            return r.item()
        if isinstance(r, _C.Tensor):
            return Tensor(_c=r)
        return r

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            # element-wise copy for matching shape
            value = value._c
        self._c[idx] = value
        self._version += 1

    def item(self):
        return self._c._flat_item(0)

    # --- factory / copy ---
    def clone(self):
        """Deep copy: new storage with same data."""
        t = _C.empty(list(self._c.sizes()))
        src = self._c.contiguous()
        for i in range(self.numel()):
            t[i] = src._flat_item(i)
        return Tensor(_c=t)

    # --- view ops (delegate to _C) ---
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        return Tensor(_c=self._c.reshape(shape))

    def reshape(self, *shape):
        return self.view(*shape)

    def t(self):
        """2D transpose."""
        assert self.dim() == 2, "t() requires 2D tensor"
        return Tensor(_c=self._c.transpose(0, 1))

    def transpose(self, dim0, dim1):
        return Tensor(_c=self._c.transpose(dim0, dim1))

    def unsqueeze(self, dim):
        sizes = list(self._c.sizes())
        if dim < 0:
            dim = len(sizes) + 1 + dim
        sizes.insert(dim, 1)
        return self.view(sizes)

    def expand_as(self, other):
        return Tensor(_c=self._c.expand(list(other._c.sizes())))

    def expand(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = list(sizes[0])
        else:
            sizes = list(sizes)
        return Tensor(_c=self._c.expand(sizes))

    def contiguous(self):
        return Tensor(_c=self._c.contiguous())

    # --- arithmetic (element-wise, returns new Tensor) ---
    def _ensure_c(self, other):
        if isinstance(other, Tensor):
            return other._c
        if isinstance(other, _C.Tensor):
            return other
        return None

    def __add__(self, other):
        c = self._ensure_c(other)
        if c is not None:
            return Tensor(_c=_C.add(self._c, c))
        # scalar
        return self._apply_scalar(lambda x: x + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        c = self._ensure_c(other)
        if c is not None:
            result = _C.add(self._c, c)
            self._c = result
        else:
            # scalar in-place
            self._c = self._apply_scalar(lambda x: x + other)._c
        self._version += 1
        return self

    def __sub__(self, other):
        c = self._ensure_c(other)
        if c is not None:
            return self._apply_binary(other, lambda a, b: a - b)
        return self._apply_scalar(lambda x: x - other)

    def __rsub__(self, other):
        return self._apply_scalar(lambda x: other - x)

    def __neg__(self):
        return self._apply_scalar(lambda x: -x)

    def __mul__(self, other):
        c = self._ensure_c(other)
        if c is not None:
            return Tensor(_c=_C.mul(self._c, c))
        # scalar
        return self._apply_scalar(lambda x: x * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._apply_scalar(lambda x: x / other)

    # --- comparison ---
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self._apply_scalar_bool(lambda x: 1.0 if x == other else 0.0)
        return self._apply_binary(other, lambda a, b: 1.0 if a == b else 0.0)

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            return self._apply_scalar_bool(lambda x: 1.0 if x != other else 0.0)
        return self._apply_binary(other, lambda a, b: 1.0 if a != b else 0.0)

    def gt(self, value):
        return self._apply_scalar_bool(lambda x: 1.0 if x > value else 0.0)

    # --- reduction ---
    def sum(self, dim=None):
        if dim is None:
            # full reduction → scalar tensor
            val = _C.sum(self._c)
            return Tensor(_c=_C.tensor([val])).view([])
        # reduce along a single dimension
        sizes = list(self._c.sizes())
        if dim < 0:
            dim += len(sizes)
        new_sizes = sizes[:dim] + sizes[dim + 1:]
        if not new_sizes:
            new_sizes = []
        result = _zeros(new_sizes if new_sizes else [1])
        src = self.contiguous()
        self._sum_along_dim(src, result, sizes, dim, new_sizes)
        if not new_sizes:
            return result.view([])
        return result

    def _sum_along_dim(self, src, result, sizes, dim, new_sizes):
        """Sum along specified dimension."""
        total = result.numel()
        ndim = len(sizes)
        for flat in range(total):
            # compute multi-index in result
            idx_r = []
            tmp = flat
            for d in range(len(new_sizes) - 1, -1, -1):
                idx_r.insert(0, tmp % new_sizes[d])
                tmp //= new_sizes[d]
            # sum along dim
            s = 0.0
            for k in range(sizes[dim]):
                # build source index
                idx_s = list(idx_r)
                idx_s.insert(dim, k)
                # read from source
                val = src
                for i in idx_s:
                    val = val[i]
                s += float(val)
            # write to result
            r = result
            for i in idx_r[:-1]:
                r = r[i]
            r[idx_r[-1]] = s

    def any(self):
        """Return True if any element is non-zero."""
        src = self._c.contiguous()
        for i in range(self.numel()):
            if src._flat_item(i) != 0:
                return True
        return False

    def float(self):
        """Already float, return self."""
        return self

    # --- matrix ops ---
    def mm(self, other):
        return Tensor(_c=_C.matmul(self._c, other._c))

    # --- element-wise ops ---
    def clamp(self, min=None, max=None):
        def _clamp(x):
            if min is not None and x < min:
                return float(min)
            if max is not None and x > max:
                return float(max)
            return x
        return self._apply_scalar(_clamp)

    # --- helpers ---
    def _apply_scalar(self, fn):
        """Apply a scalar function to every element, return new Tensor."""
        src = self._c.contiguous()
        n = self.numel()
        result = _C.empty(list(self._c.sizes()))
        for i in range(n):
            result[i] = fn(src._flat_item(i))
        return Tensor(_c=result)

    def _apply_scalar_bool(self, fn):
        return self._apply_scalar(fn)

    def _apply_binary(self, other, fn):
        """Apply binary fn element-wise between self and other."""
        if isinstance(other, Tensor):
            other_c = other._c
        else:
            other_c = other._c if hasattr(other, '_c') else None
        src_a = self._c.contiguous()
        if other_c is not None:
            src_b = other_c.contiguous()
        n = self.numel()
        result = _C.empty(list(self._c.sizes()))
        for i in range(n):
            a = src_a._flat_item(i)
            b = src_b._flat_item(i) if other_c is not None else float(other)
            result[i] = fn(a, b)
        return Tensor(_c=result)

    def __repr__(self):
        return repr(self._c)

    def __len__(self):
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._c.sizes()[0]

    def __float__(self):
        if self.numel() == 1:
            return float(self._c._flat_item(0))
        raise ValueError("only one element tensors can be converted to Python scalars")

    def __bool__(self):
        if self.numel() == 1:
            return self._c._flat_item(0) != 0.0
        raise ValueError("bool on multi-element tensor is ambiguous")


# --- Module-level factory helpers ---

def _zeros(shape):
    return Tensor(_c=_C.empty(shape))


def _ones(shape):
    return Tensor(_c=_C.ones(shape))


def zeros(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)
    return _zeros(shape)


def ones(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)
    return _ones(shape)


def randn(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)
    n = 1
    for s in shape:
        n *= s
    t = _C.empty(shape)
    for i in range(n):
        # Box-Muller transform
        u1 = _rng.random()
        u2 = _rng.random()
        z = math.sqrt(-2.0 * math.log(u1 + 1e-30)) * math.cos(2.0 * math.pi * u2)
        t[i] = z
    return Tensor(_c=t)


def FloatTensor(data):
    """Create a Tensor from a list/nested list, like torch.FloatTensor(...)."""
    if isinstance(data, (list, tuple)):
        return Tensor(data)
    raise TypeError(f"FloatTensor expects list, got {type(data)}")
