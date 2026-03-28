"""
Autograd Engine — C++ 主引擎 + Python 兼容层
"""

import sys
import os
import weakref
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _C
import _nn_C


_python_grad_tensors = []


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
        _python_grad_tensors.append(weakref.ref(out))

def iter_mixed_roots():
    """返回已收到 C++ 梯度、需要继续走 Python backward 的根张量。"""
    roots = []
    alive = []
    for ref in _python_grad_tensors:
        t = ref()
        if t is None:
            continue
        alive.append(ref)
        if t.__dict__.get('_py_grad_fn') is None:
            continue
        grad = t.grad
        if grad is None:
            continue
        roots.append((t, grad))
    _python_grad_tensors[:] = alive
    return roots


def backward(loss_tensor, grad_output=None):
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
    grad_map[id(loss_tensor)] = grad_output if grad_output is not None else _ones_like(loss_tensor)

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
                t._c.set_requires_grad(True)
                t._c.accumulate_grad(g._c)


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
