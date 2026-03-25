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
        self.requires_grad = requires_grad
        self.grad = None
        self._dtype = float32  # 默认 float32
        self._grad_fn = None   # autograd graph 节点

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
    def shape(self):
        return tuple(self._c.sizes())

    @property
    def data(self):
        """兼容 Variable 接口：Tensor.data 返回自身"""
        return self

    @property
    def _version(self):
        """版本计数器，用于 autograd in-place 检测"""
        return self._c._version()

    # ---- Autograd 支持 ----

    def backward(self):
        """执行反向传播"""
        from torch.autograd_engine import backward
        backward(self)

    def detach(self):
        """返回共享数据但不跟踪梯度的新 Tensor"""
        t = Tensor(self._c)
        t.requires_grad = False
        t.grad = None
        t._grad_fn = None
        return t

    def numpy(self):
        """转换为 numpy 数组"""
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
        """去掉大小为 1 的维度"""
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
        """沿指定维度做 slice"""
        sizes = self.size()
        n = sizes[dim]
        start, stop, step = s.indices(n)
        if step != 1:
            raise RuntimeError("step != 1 not supported")
        result = Tensor(_nn_C.clone(self._c))
        # 使用 C++ slice
        from torch.tensor import Tensor as T
        sliced = self._c
        # 需要用 native::slice
        import _C as _c_mod
        # 手动构造 slice view
        sizes_list = list(self._c.sizes())
        strides_list = list(self._c.strides())
        new_offset = self._c.storage_offset() + start * strides_list[dim]
        new_sizes = list(sizes_list)
        new_sizes[dim] = stop - start
        # 通过 reshape 和 flat_get/set 实现
        ct = self._c if self._c.is_contiguous() else self._c.contiguous()
        result_c = _C.empty(new_sizes)
        # 通用 N-dim slice copy
        n_total = 1
        for s_val in new_sizes:
            n_total *= s_val
        src_strides = []
        stride = 1
        for i in range(len(sizes_list) - 1, -1, -1):
            src_strides.insert(0, stride)
            stride *= sizes_list[i]

        for flat_idx in range(n_total):
            # 将 flat_idx 转为 new_sizes 坐标
            coords = []
            rem = flat_idx
            for d in range(len(new_sizes)):
                coords.append(rem // (n_total // new_sizes[d] if d == 0 else 1))
            # 简化: 用递归方式
            coords = []
            rem = flat_idx
            for d in range(len(new_sizes) - 1, -1, -1):
                coords.insert(0, rem % new_sizes[d])
                rem //= new_sizes[d]
            # 映射到源坐标
            src_coords = list(coords)
            src_coords[dim] += start
            # 计算源 flat idx
            src_flat = 0
            for d in range(len(sizes_list)):
                src_flat += src_coords[d] * src_strides[d]
            result_c.flat_set(flat_idx, ct.flat_get(src_flat))

        t = Tensor(result_c)
        # 传播 grad
        if self.requires_grad or self._grad_fn is not None:
            from torch.autograd_engine import record
            record([t], [self], lambda go: [_zeros_like_and_scatter(self, go[0], dim, start, stop)], "slice")
        return t

    def _multi_index(self, idx):
        """处理多维索引，支持 int 和 slice 混合"""
        sizes = self.size()
        # 如果全是 int, 用旧逻辑
        if all(isinstance(i, int) for i in idx):
            result = self._c
            for i in idx:
                result = result[i]
            t = Tensor(result)
            if t.dim() == 0:
                return t.item()
            return t

        # 处理含 slice 的索引
        current = self
        dim_offset = 0
        for dim_idx, index in enumerate(idx):
            actual_dim = dim_idx - dim_offset
            if isinstance(index, int):
                # int 索引: 降维
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
                # None = unsqueeze
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

    # ---- 算术运算 ----

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(_nn_C.broadcast_add(self._c, other._c))
            # 传播 autograd
            if (self.requires_grad or getattr(self, '_grad_fn', None)) or \
               (other.requires_grad or getattr(other, '_grad_fn', None)):
                from torch.autograd_engine import record
                saved_self_shape = self.size()
                saved_other_shape = other.size()
                def backward_fn(grad_outputs):
                    g = grad_outputs[0]
                    ga = _reduce_grad(g, saved_self_shape)
                    gb = _reduce_grad(g, saved_other_shape)
                    return [ga, gb]
                record([result], [self, other], backward_fn, "add")
            return result
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
        result = Tensor(self._c.reshape(shape))
        # 传播 autograd 信息
        if self.requires_grad or self._grad_fn is not None:
            from torch.autograd_engine import record
            saved_self = self
            original_shape = self.size()
            def backward_fn(grad_outputs):
                g = grad_outputs[0]
                return [g.view(original_shape)]
            record([result], [self], backward_fn, "view")
        return result

    def reshape(self, *shape):
        return self.view(*shape)

    def unsqueeze(self, dim):
        """在指定维度插入大小为 1 的维度"""
        sizes = list(self._c.sizes())
        if dim < 0:
            dim = len(sizes) + 1 + dim
        new_sizes = sizes[:dim] + [1] + sizes[dim:]
        # 直接调用 view 来保持 autograd 链
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


def tensor(data, dtype=None):
    """从嵌套 list/numpy array 创建 Tensor，支持 dtype"""
    import numpy as np

    if isinstance(data, np.ndarray):
        flat = data.flatten().tolist()
        shape = list(data.shape)
    elif isinstance(data, (list, tuple)):
        # 展平嵌套 list
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
    """生成 [low, high) 范围内的随机整数张量"""
    if isinstance(size, (list, tuple)):
        size = list(size)
    c = _nn_C.randint(low, high, size)
    t = Tensor(c)
    t._dtype = long
    return t


def argmax(input_tensor, dim=None):
    """沿指定维度求最大值索引"""
    if dim is None:
        # 全局 argmax
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
# 内部辅助函数 (autograd 用)
# ============================================================

def _reduce_grad(grad, target_shape):
    """将 grad 归约到 target_shape (处理广播的反向)"""
    g_shape = grad.size()
    t_shape = list(target_shape)

    if g_shape == t_shape:
        return grad

    # 补齐维度
    ndim_diff = len(g_shape) - len(t_shape)
    t_shape_padded = [1] * ndim_diff + t_shape

    # 沿广播维度求和
    result = grad
    for i in range(len(g_shape)):
        if t_shape_padded[i] == 1 and g_shape[i] > 1:
            result = Tensor(_nn_C.sum_dim(result._c, i, True))

    # 去掉多余的前导维度
    if ndim_diff > 0:
        result = result.view(t_shape)

    return result


def _zeros_like_and_scatter(original, grad_slice, dim, start, stop):
    """创建与 original 同形状的零张量，在 [start:stop] 位置填入 grad_slice"""
    result = Tensor(_C.empty(original.size()))
    sizes = original.size()
    slice_sizes = grad_slice.size()

    n_total = grad_slice.numel()
    # 计算 strides
    src_strides = []
    stride = 1
    for i in range(len(sizes) - 1, -1, -1):
        src_strides.insert(0, stride)
        stride *= sizes[i]

    gs = grad_slice._c if grad_slice._c.is_contiguous() else grad_slice._c.contiguous()

    for flat_idx in range(n_total):
        coords = []
        rem = flat_idx
        for d in range(len(slice_sizes) - 1, -1, -1):
            coords.insert(0, rem % slice_sizes[d])
            rem //= slice_sizes[d]
        # 映射到原始坐标
        src_coords = list(coords)
        src_coords[dim] += start
        src_flat = 0
        for d in range(len(sizes)):
            src_flat += src_coords[d] * src_strides[d]
        result._c.flat_set(src_flat, gs.flat_get(flat_idx))

    return result
