"""
Python Tensor 包装类
包装 _C.Tensor，提供 Python 风格的接口
autograd 完全在 C++ 侧实现，Python 零开销
"""

import sys
import os
import math
import random

# 导入 C++ 扩展模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _C
import _nn_C


# dtype 常量
class _DType:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"torch.{self.name}"

float32 = _DType("float32")
long = _DType("long")


class Tensor:
    """Python 侧的 Tensor 包装类，包装 _C.Tensor"""

    def __init__(self, c_tensor=None, requires_grad=False):
        if c_tensor is None:
            self._c = _C.Tensor([0])
        elif isinstance(c_tensor, _C.Tensor):
            self._c = c_tensor
        elif isinstance(c_tensor, Tensor):
            self._c = c_tensor._c
        else:
            raise TypeError(f"Tensor: unsupported init type {type(c_tensor)}")
        if requires_grad:
            self._c.set_requires_grad(True)
        self._dtype = float32  # 默认 float32

    # ---- 基本属性 ----

    @property
    def requires_grad(self):
        return self._c.requires_grad()

    @requires_grad.setter
    def requires_grad(self, value):
        self._c.set_requires_grad(value)

    @property
    def grad(self):
        """获取梯度（从 C++ TensorImpl 读取）"""
        g = self._c.grad()
        if g is None:
            return None
        return Tensor(g)

    @grad.setter
    def grad(self, value):
        # 兼容旧代码中 self.grad = None 的写法
        if value is not None:
            raise RuntimeError("direct grad assignment not supported in new system")

    @property
    def _grad_fn(self):
        """兼容旧代码：返回 creator 或 Python GradFn"""
        # 先检查 Python 侧的 _grad_fn (System B 兼容)
        pfn = self.__dict__.get('_py_grad_fn')
        if pfn is not None:
            return pfn
        c = self._c.get_creator()
        return c if c is not None else None

    @_grad_fn.setter
    def _grad_fn(self, value):
        """兼容 System B: 允许 Python 侧设置 _grad_fn"""
        self.__dict__['_py_grad_fn'] = value

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
        return Tensor(_nn_C.clone(self._c))

    @property
    def shape(self):
        return tuple(self._c.sizes())

    @property
    def data(self):
        """兼容 Variable 接口：Tensor.data 返回自身"""
        return self

    @property
    def _version(self):
        return self._c._version()

    # ---- Autograd 支持 ----

    def backward(self):
        """执行反向传播"""
        if self._c.has_creator():
            # C++ backward — 处理主图
            _C.autograd_backward(self._c)
            # 之后检查是否有 Python _grad_fn 节点需要继续传播
            # (例如 RNN 的 output 在 C++ 被视为叶子，但有 Python grad_fn)
            self._continue_python_backward()
        elif self.__dict__.get('_py_grad_fn') is not None:
            from torch.autograd_engine import backward as py_backward
            py_backward(self)
        else:
            raise RuntimeError("backward: tensor has no grad_fn")

    def _continue_python_backward(self):
        """
        C++ backward 完成后，检查有 Python _grad_fn 的张量是否收到了 C++ grad。
        如果是，从那些张量继续 Python backward。
        """
        from torch.autograd_engine import backward as py_backward
        from torch.autograd_engine import iter_mixed_roots

        seen = set()
        for root, grad in iter_mixed_roots():
            root_id = id(root)
            if root_id in seen:
                continue
            seen.add(root_id)
            py_backward(root, grad_output=grad)

    def zero_grad(self):
        self._c.zero_grad()

    def detach(self):
        """返回共享数据但不跟踪梯度的新 Tensor"""
        t = Tensor(self._c)
        # detach: 不设置 requires_grad，不传播 creator
        # 注意: 共享底层数据但 C++ 侧的 creator 仍在原 TensorImpl 上
        # 需要创建新的 TensorImpl 来真正 detach
        sizes = list(self._c.sizes())
        result = _C.empty(sizes)
        n = self.numel()
        src = self._c if self._c.is_contiguous() else self._c.contiguous()
        for i in range(n):
            result.flat_set(i, src.flat_get(i))
        return Tensor(result)

    def numpy(self):
        import numpy as np
        sizes = list(self._c.sizes())
        n = self.numel()
        c = self._c if self._c.is_contiguous() else self._c.contiguous()
        data = [c.flat_get(i) for i in range(n)]
        if self._dtype is long:
            arr = np.array(data, dtype=np.int64).reshape(sizes) if sizes else np.array(data[0], dtype=np.int64)
        else:
            arr = np.array(data, dtype=np.float32).reshape(sizes) if sizes else np.array(data[0], dtype=np.float32)
        return arr

    def squeeze(self, dim=None):
        sizes = list(self._c.sizes())
        if dim is not None:
            if dim < 0:
                dim = len(sizes) + dim
            if sizes[dim] == 1:
                new_sizes = sizes[:dim] + sizes[dim + 1:]
                if not new_sizes:
                    new_sizes = [1]
                return self.view(new_sizes)
            return self
        new_sizes = [s for s in sizes if s != 1]
        if not new_sizes:
            new_sizes = [1]
        return self.view(new_sizes)

    # ---- 索引 ----

    def __getitem__(self, idx):
        if isinstance(idx, int):
            result = Tensor(self._c[idx])
            if result.dim() == 0:
                return result.item()
            return result
        elif isinstance(idx, slice):
            return self._slice_dim(0, idx)
        elif isinstance(idx, tuple):
            return self._multi_index(idx)
        raise TypeError(f"unsupported index type: {type(idx)}")

    def _slice_dim(self, dim, s):
        sizes = self.size()
        n = sizes[dim]
        start, stop, step = s.indices(n)
        if step != 1:
            raise RuntimeError("step != 1 not supported")
        return Tensor(_C.autograd_slice(self._c, dim, start, stop))

    def _multi_index(self, idx):
        sizes = self.size()
        if all(isinstance(i, int) for i in idx):
            result = self._c
            for i in idx:
                result = result[i]
            t = Tensor(result)
            if t.dim() == 0:
                return t.item()
            return t

        current = self
        dim_offset = 0
        for dim_idx, index in enumerate(idx):
            actual_dim = dim_idx - dim_offset
            if isinstance(index, int):
                sizes_list = current.size()
                n = sizes_list[actual_dim]
                if index < 0:
                    index += n
                current = current._slice_dim(actual_dim, slice(index, index + 1))
                current = current.squeeze(actual_dim)
                dim_offset += 1
            elif isinstance(index, slice):
                current = current._slice_dim(actual_dim, index)
            elif index is None:
                current = current.unsqueeze(actual_dim)
                dim_offset -= 1
        return current

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

    # ---- 算术运算（全部委托 C++ autograd） ----

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(_C.autograd_add(self._c, other._c))
        elif isinstance(other, (int, float)):
            return Tensor(_C.autograd_add_scalar(self._c, float(other)))
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
            return Tensor(_C.autograd_sub(self._c, other._c))
        elif isinstance(other, (int, float)):
            return Tensor(_C.autograd_sub_scalar(self._c, float(other)))
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(_C.autograd_add_scalar(_C.autograd_neg(self._c), float(other)))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(_C.autograd_mul(self._c, other._c))
        elif isinstance(other, (int, float)):
            return Tensor(_C.autograd_mul_scalar(self._c, float(other)))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(_C.autograd_div_scalar(self._c, float(other)))
        elif isinstance(other, Tensor):
            return Tensor(_C.autograd_div(self._c, other._c))
        return NotImplemented

    def __neg__(self):
        return Tensor(_C.autograd_neg(self._c))

    # ---- 矩阵运算 ----

    def mm(self, other):
        return Tensor(_C.autograd_mm(self._c, other._c))

    def matmul(self, other):
        if self.dim() == 2 and other.dim() == 2:
            return self.mm(other)
        elif self.dim() == 3 and other.dim() == 2:
            B, M, K = self.size()
            N = other.size(1)
            return self.view(B * M, K).mm(other).view(B, M, N)
        elif self.dim() >= 3 and other.dim() >= 2:
            return Tensor(_C.autograd_batched_matmul(self._c, other._c))
        raise RuntimeError(
            f"matmul: unsupported dims {self.dim()} x {other.dim()}")

    def t(self):
        if self.dim() != 2:
            raise RuntimeError("t() expects a 2D tensor")
        return self.transpose(0, 1)

    def transpose(self, dim0, dim1):
        return Tensor(_C.autograd_transpose(self._c, dim0, dim1))

    # ---- 变形 ----

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        return Tensor(_C.autograd_view(self._c, shape))

    def reshape(self, *shape):
        return self.view(*shape)

    def unsqueeze(self, dim):
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
        return Tensor(_C.autograd_expand(self._c, sizes))

    def expand_as(self, other):
        return self.expand(other.size())

    def contiguous(self):
        return Tensor(self._c.contiguous())

    # ---- 激活函数 ----

    def relu(self):
        return Tensor(_C.autograd_relu(self._c))

    def tanh(self):
        return Tensor(_C.autograd_tanh(self._c))

    # ---- 比较 ----

    def gt(self, value):
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

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(_C.autograd_sum(self._c))
        else:
            if dim < 0:
                dim = self.dim() + dim
            return Tensor(_C.autograd_sum_dim(self._c, dim, keepdim))

    def any(self):
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
        return self

    def __float__(self):
        return float(self.item())

    def __bool__(self):
        return bool(self.item())

    def __repr__(self):
        return f"Tensor({self._c.__repr__()})"

    # ---- 内部辅助方法 ----

    def _read_elem(self, flat_idx):
        return self._c.flat_get(flat_idx)

    def _apply_scalar(self, scalar, op):
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
        sizes = list(self._c.sizes())
        result = _C.empty(sizes)
        n = self.numel()
        for i in range(n):
            result.flat_set(i, op(self._c.flat_get(i), other._c.flat_get(i)))
        return Tensor(result)


# ============================================================
# 工厂函数
# ============================================================

def FloatTensor(data):
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
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = list(shape[0])
    else:
        shape = list(shape)

    n = 1
    for s in shape:
        n *= s

    result = _C.empty(shape)
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


def tensor(data, dtype=None):
    import numpy as np

    if isinstance(data, np.ndarray):
        flat = data.flatten().tolist()
        shape = list(data.shape)
    elif isinstance(data, (list, tuple)):
        arr = np.array(data)
        flat = arr.flatten().tolist()
        shape = list(arr.shape)
    elif isinstance(data, (int, float)):
        flat = [float(data)]
        shape = [1]
    else:
        raise TypeError(f"tensor: unsupported data type {type(data)}")

    c = _C.empty(shape)
    for i, v in enumerate(flat):
        c.flat_set(i, float(v))

    t = Tensor(c)
    if dtype is not None:
        t._dtype = dtype
    return t


def randint(low, high, size):
    if isinstance(size, (list, tuple)):
        size = list(size)
    c = _nn_C.randint(low, high, size)
    t = Tensor(c)
    t._dtype = long
    return t


def argmax(input_tensor, dim=None):
    if dim is None:
        n = input_tensor.numel()
        max_val = input_tensor._c.flat_get(0)
        max_idx = 0
        for i in range(1, n):
            v = input_tensor._c.flat_get(i)
            if v > max_val:
                max_val = v
                max_idx = i
        return Tensor(_C.Tensor([1], float(max_idx)))
    result = _nn_C.argmax(input_tensor._c, dim)
    t = Tensor(result)
    t._dtype = long
    return t


# ============================================================
# 内部辅助函数 (保留用于兼容)
# ============================================================

def _reduce_grad(grad, target_shape):
    """将 grad 归约到 target_shape (处理广播的反向)"""
    g_shape = grad.size()
    t_shape = list(target_shape)

    if g_shape == t_shape:
        return grad

    ndim_diff = len(g_shape) - len(t_shape)
    t_shape_padded = [1] * ndim_diff + t_shape

    result = grad
    for i in range(len(g_shape)):
        if t_shape_padded[i] == 1 and g_shape[i] > 1:
            result = Tensor(_nn_C.sum_dim(result._c, i, True))

    if ndim_diff > 0:
        result = result.view(t_shape)

    return result
