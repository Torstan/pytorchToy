"""
Autograd Engine — C++ 主引擎 + Python 兼容层
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _C
import _nn_C


class no_grad:
    """Context manager: 禁用梯度记录"""

    def __enter__(self):
        self._prev = _C.is_grad_enabled()
        _C.set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        _C.set_grad_enabled(self._prev)


class GradFn:
    """Python backward 节点（兼容层）"""
    __slots__ = ['inputs', 'backward_fn', 'name']

    def __init__(self, inputs, backward_fn, name=""):
        self.inputs = inputs
        self.backward_fn = backward_fn
        self.name = name


def record(outputs, inputs, backward_fn, name=""):
    """
    记录操作到 Python 计算图（兼容 RNN 等尚未迁移的模块）。
    """
    if not _C.is_grad_enabled():
        return
    needs_grad = False
    for inp in inputs:
        if getattr(inp, 'requires_grad', False):
            needs_grad = True
            break
        if hasattr(inp, '_grad_fn') and inp._grad_fn is not None:
            needs_grad = True
            break
    if not needs_grad:
        return

    grad_fn = GradFn(inputs, backward_fn, name)
    for out in outputs:
        out._grad_fn = grad_fn
        out.requires_grad = True


def backward(loss_tensor):
    """
    Python backward — 遍历 Python _grad_fn 图。
    用于 RNN 等仍使用 record() 的模块。
    """
    from torch.tensor import Tensor

    visited_tensors = set()
    order = []

    def _topo_sort(t):
        tid = id(t)
        if tid in visited_tensors:
            return
        visited_tensors.add(tid)
        gf = t.__dict__.get('_py_grad_fn')
        if gf is not None:
            for inp in gf.inputs:
                _topo_sort(inp)
        order.append(t)

    _topo_sort(loss_tensor)

    grad_map = {}
    grad_map[id(loss_tensor)] = _ones_like(loss_tensor)

    for t in reversed(order):
        gf = t.__dict__.get('_py_grad_fn')
        if gf is None:
            continue

        grad_out = grad_map.get(id(t))
        if grad_out is None:
            continue

        grad_inputs = gf.backward_fn([grad_out])
        if not isinstance(grad_inputs, (list, tuple)):
            grad_inputs = [grad_inputs]

        for inp, grad in zip(gf.inputs, grad_inputs):
            if grad is None:
                continue
            old = grad_map.get(id(inp))
            if old is None:
                grad_map[id(inp)] = grad
            else:
                grad_map[id(inp)] = _add_tensors(old, grad)

    # 将梯度写入叶子参数
    for t in order:
        if getattr(t, 'requires_grad', False) and id(t) in grad_map:
            if t.__dict__.get('_py_grad_fn') is None:
                g = grad_map[id(t)]
                # 累加到 C++ TensorImpl 的 grad
                gc = g._c if g._c.is_contiguous() else g._c.contiguous()
                t._c.set_requires_grad(True)
                # 用 Python 手动累加
                sizes = list(gc.sizes())
                n = gc.numel()
                flat_data = [gc.flat_get(i) for i in range(n)]
                # 调用 C++ accumulate_grad 的替代方案
                if not t._c.has_grad():
                    # 首次: 创建零张量
                    pass  # accumulate_grad 内部处理
                # 直接用 C++ flat_set 实现
                if not t._c.has_grad():
                    # 初始化 grad
                    zero = _C.empty(sizes)
                    for i in range(n):
                        zero.flat_set(i, flat_data[i])
                    # 不能直接设置 grad storage，用 workaround
                    # 先用 Python 级别的 grad 属性
                    t.__dict__['_py_grad'] = g
                else:
                    old_g = t.__dict__.get('_py_grad')
                    if old_g is not None:
                        t.__dict__['_py_grad'] = _add_tensors(old_g, g)
                    else:
                        t.__dict__['_py_grad'] = g


# ---- 辅助函数 ----

def _ones_like(t):
    from torch.tensor import Tensor
    return Tensor(_C.ones(t.size()))

def _zeros_like(t):
    from torch.tensor import Tensor
    return Tensor(_C.empty(t.size()))

def _add_tensors(a, b):
    from torch.tensor import Tensor
    return Tensor(_nn_C.broadcast_add(a._c, b._c))
