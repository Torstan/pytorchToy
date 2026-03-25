"""
Tape-based Autograd Engine (Graph-based)

每个 Tensor 记录产生它的操作 (_grad_fn)，
backward 时沿 _grad_fn 链反向遍历计算梯度。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _C
import _nn_C


_enabled = True  # 全局开关


class GradFn:
    """一次操作的反向传播节点"""
    __slots__ = ['inputs', 'backward_fn', 'name']

    def __init__(self, inputs, backward_fn, name=""):
        """
        inputs: list of (Tensor or Parameter) — 前向输入（需要梯度的和不需要的都包含）
        backward_fn: callable (grad_outputs) → list of grad_inputs
        name: 调试用名称
        """
        self.inputs = inputs
        self.backward_fn = backward_fn
        self.name = name


def record(outputs, inputs, backward_fn, name=""):
    """
    记录一次操作到计算图。
    将 GradFn 设置到每个 output 的 _grad_fn 上。
    """
    if not _enabled:
        return

    # 检查是否有任何输入需要梯度
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
    从 loss 反向遍历计算图，计算并累积梯度。
    使用拓扑排序确保正确的梯度流动。
    """
    from torch.tensor import Tensor

    # 收集所有需要遍历的节点 (BFS)
    visited_tensors = set()
    order = []  # 反向拓扑序

    def _topo_sort(t):
        tid = id(t)
        if tid in visited_tensors:
            return
        visited_tensors.add(tid)
        gf = getattr(t, '_grad_fn', None)
        if gf is not None:
            for inp in gf.inputs:
                _topo_sort(inp)
        order.append(t)

    _topo_sort(loss_tensor)

    # 梯度表: tensor id → grad Tensor
    grad_map = {}
    grad_map[id(loss_tensor)] = _ones_like(loss_tensor)

    # 反向遍历 (从 loss 到参数)
    for t in reversed(order):
        gf = getattr(t, '_grad_fn', None)
        if gf is None:
            continue

        grad_out = grad_map.get(id(t))
        if grad_out is None:
            continue

        # 调用 backward 函数
        grad_inputs = gf.backward_fn([grad_out])
        if not isinstance(grad_inputs, (list, tuple)):
            grad_inputs = [grad_inputs]

        # 累积输入梯度
        for inp, grad in zip(gf.inputs, grad_inputs):
            if grad is None:
                continue
            old = grad_map.get(id(inp))
            if old is None:
                grad_map[id(inp)] = grad
            else:
                grad_map[id(inp)] = _add_tensors(old, grad)

    # 将梯度写入参数的 .grad 属性
    for t in order:
        if getattr(t, 'requires_grad', False) and id(t) in grad_map:
            # 只给叶子节点 (没有 _grad_fn 的 requires_grad 张量, 即 Parameter) 设置 grad
            if getattr(t, '_grad_fn', None) is None:
                g = grad_map[id(t)]
                if t.grad is None:
                    t.grad = g
                else:
                    t.grad = _add_tensors(t.grad, g)


class no_grad:
    """Context manager: 禁用梯度记录"""

    def __enter__(self):
        global _enabled
        self._prev = _enabled
        _enabled = False
        return self

    def __exit__(self, *args):
        global _enabled
        _enabled = self._prev


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
