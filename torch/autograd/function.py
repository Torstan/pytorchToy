"""
Function 基类 — 用户自定义 autograd 函数继承此类
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C


class Function:
    """
    autograd Function 基类

    用户继承此类并实现 forward() 和 backward():
        class MyFunc(Function):
            def forward(self, x):
                self.save_for_backward(x)
                return x * x
            def backward(self, grad_output):
                x, = self.saved_tensors
                return 2 * x * grad_output

    调用方式: y = MyFunc()(x)
    """

    def __init__(self):
        self._saved_tensors = ()
        self._dirty_tensors = set()

    def __call__(self, *inputs):
        from torch.autograd.variable import Variable
        from torch.tensor import Tensor

        # 判断是否有 requires_grad 的输入
        any_requires_grad = False
        any_volatile = False

        for inp in inputs:
            if isinstance(inp, Variable):
                if inp.requires_grad:
                    any_requires_grad = True
                if inp.volatile:
                    any_volatile = True

        # 从 Variable 中提取 raw Tensor
        raw_tensors = []
        for inp in inputs:
            if isinstance(inp, Variable):
                raw_tensors.append(inp.data)
            elif isinstance(inp, Tensor):
                raw_tensors.append(inp)
            else:
                raw_tensors.append(inp)

        # 调用用户的 forward
        output = self.forward(*raw_tensors)

        # 对 mark_dirty 标记的张量递增版本，并更新 saved_tensors 中的版本号
        if self._dirty_tensors:
            for t in raw_tensors:
                if hasattr(t, '_c') and id(t) in self._dirty_tensors:
                    t._c._bump_version()
            # 更新 saved_tensors 中 dirty 张量的期望版本号
            updated = []
            for item in self._saved_tensors:
                t, ver = item
                if id(t) in self._dirty_tensors:
                    updated.append((t, t._version if hasattr(t, '_version') else 0))
                else:
                    updated.append(item)
            self._saved_tensors = updated

        # 确保输出是 Tensor
        if not isinstance(output, Tensor):
            output = Tensor(output) if hasattr(output, '_c') else output

        # 包装为 Variable
        if any_volatile:
            # volatile 模式：不构建计算图
            result = Variable(output, requires_grad=False, volatile=True)
            return result

        result = Variable(output, requires_grad=any_requires_grad)

        if any_requires_grad:
            # 创建 PyFunction 节点（C++ 桥接）
            py_fn = _C.PyFunction(self)
            py_fn.num_inputs = len(inputs)
            py_fn.requires_grad = True

            # 连接计算图
            for inp in inputs:
                if isinstance(inp, Variable):
                    if inp._creator_fn is not None:
                        # 非叶子变量：链接到上游函数
                        py_fn.add_previous_function(inp._creator_fn, inp._output_index)
                    else:
                        # 叶子变量：记录 VariableImpl
                        py_fn.add_leaf_variable(inp._impl)
                else:
                    # 非 Variable 输入，不需要梯度
                    py_fn.add_leaf_variable(_C.VariableImpl(_C.Tensor([1], 0.0), False, False))

            result._creator_fn = py_fn
            result.creator = self

        return result

    def save_for_backward(self, *tensors):
        """保存张量用于 backward，同时记录版本号"""
        self._saved_tensors = []
        for t in tensors:
            version = t._version if hasattr(t, '_version') else 0
            self._saved_tensors.append((t, version))

    @property
    def saved_tensors(self):
        """获取保存的张量，检查是否被 in-place 修改"""
        result = []
        for t, expected_version in self._saved_tensors:
            current_version = t._version if hasattr(t, '_version') else 0
            if current_version != expected_version:
                raise RuntimeError(
                    "one of the variables needed for gradient computation has been "
                    "modified by an inplace operation"
                )
            result.append(t)
        return tuple(result)

    def mark_dirty(self, *tensors):
        """声明 forward 中被 in-place 修改的张量，使版本检查跳过这些张量"""
        for t in tensors:
            self._dirty_tensors.add(id(t))

    def _do_backward(self, grad_outputs_list):
        """
        C++ Engine 回调入口
        接收 _C.Tensor 列表，转为 Python Tensor，调用 backward，转回 _C.Tensor
        """
        from torch.tensor import Tensor

        # 将 _C.Tensor 包装为 Python Tensor
        py_grads = []
        for g in grad_outputs_list:
            if isinstance(g, _C.Tensor):
                py_grads.append(Tensor(g))
            elif isinstance(g, Tensor):
                py_grads.append(g)
            else:
                py_grads.append(g)

        # 调用用户定义的 backward
        result = self.backward(*py_grads)

        # 将结果转回 _C.Tensor
        if isinstance(result, tuple):
            return tuple(r._c if isinstance(r, Tensor) else r for r in result)
        elif isinstance(result, Tensor):
            return (result._c,)
        else:
            return (result,)

    def forward(self, *inputs):
        raise NotImplementedError("Subclass must implement forward()")

    def backward(self, *grad_outputs):
        raise NotImplementedError("Subclass must implement backward()")
