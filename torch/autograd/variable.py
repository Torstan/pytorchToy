"""
Variable — 包装 Tensor，追踪计算图用于自动微分
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C


class Variable:
    """
    Variable 包装 Tensor，支持自动微分。

    属性:
        data: 底层 Tensor
        grad: 梯度 Variable（backward 后设置）
        requires_grad: 是否需要计算梯度
        volatile: 推断模式（不构建计算图）
        creator: 产生此变量的 Function 实例（叶子变量为 None）
    """

    def __init__(self, data, requires_grad=False, volatile=False):
        from torch.tensor import Tensor

        if isinstance(data, Tensor):
            self.data = data
        elif isinstance(data, _C.Tensor):
            self.data = Tensor(data)
        else:
            raise TypeError(f"Variable data must be Tensor, got {type(data)}")

        self.requires_grad = requires_grad
        self.volatile = volatile
        self.creator = None  # Python Function 实例

        # C++ 端 VariableImpl（用于引擎的梯度累加）
        self._impl = _C.VariableImpl(self.data._c, requires_grad, volatile)

        # C++ 端 creator 函数节点
        self._creator_fn = None  # shared_ptr<AutogradFunction>
        self._output_index = 0

        # Hooks
        self._hooks = []

    @property
    def grad(self):
        """从 C++ VariableImpl 读取梯度"""
        from torch.tensor import Tensor
        if self._impl.grad_defined:
            return Variable(Tensor(self._impl.grad), requires_grad=False)
        return None

    @grad.setter
    def grad(self, value):
        """设置梯度（一般不直接调用）"""
        pass  # 梯度由 C++ 引擎管理

    def backward(self, grad_output=None, retain_variables=False):
        """执行反向传播"""
        from torch.tensor import Tensor

        if not self.requires_grad:
            return

        if self._creator_fn is None:
            return

        # 默认 grad_output = 全 1（与 data 同形状）
        if grad_output is None:
            sizes = self.data.size()
            grad_tensor = _C.ones(sizes)
        elif isinstance(grad_output, Tensor):
            grad_tensor = grad_output._c
        elif isinstance(grad_output, _C.Tensor):
            grad_tensor = grad_output
        else:
            raise TypeError(f"grad_output type not supported: {type(grad_output)}")

        # 调用 C++ 引擎执行反向传播
        _C.engine_backward(self._creator_fn, grad_tensor, retain_variables)

    def register_hook(self, hook):
        """注册梯度 hook"""
        from torch.tensor import Tensor
        self._hooks.append(hook)

        # 在 C++ 端注册 hook，包装 _C.Tensor → Python Tensor
        def c_hook(grad_tensor):
            py_grad = Tensor(grad_tensor)
            hook(py_grad)

        self._impl.add_hook(c_hook)

    def sum(self):
        """sum 操作，返回新 Variable 并构建计算图"""
        from torch.tensor import Tensor

        # Forward: 计算 sum
        val = _C.sum(self.data._c)
        result_tensor = Tensor(_C.Tensor([1], val))

        if self.volatile:
            return Variable(result_tensor, requires_grad=False, volatile=True)

        result = Variable(result_tensor, requires_grad=self.requires_grad)

        if self.requires_grad:
            sum_fn = _C.SumBackward(self.data.size())
            sum_fn.num_inputs = 1
            sum_fn.requires_grad = True

            if self._creator_fn is not None:
                sum_fn.add_previous_function(self._creator_fn, self._output_index)
            else:
                sum_fn.add_leaf_variable(self._impl)

            result._creator_fn = sum_fn

        return result

    def __mul__(self, other):
        """Variable * Variable 或 Variable * scalar"""
        if isinstance(other, Variable):
            return self._mul_variable(other)
        elif isinstance(other, (int, float)):
            return self._mul_scalar(float(other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def _mul_variable(self, other):
        from torch.tensor import Tensor

        result_tensor = self.data * other.data

        any_volatile = self.volatile or other.volatile
        any_requires_grad = self.requires_grad or other.requires_grad

        if any_volatile:
            return Variable(result_tensor, requires_grad=False, volatile=True)

        result = Variable(result_tensor, requires_grad=any_requires_grad)

        if any_requires_grad:
            mul_fn = _C.MulBackward(self.data._c, other.data._c)
            mul_fn.num_inputs = 2
            mul_fn.requires_grad = True

            if self._creator_fn is not None:
                mul_fn.add_previous_function(self._creator_fn, self._output_index)
            else:
                mul_fn.add_leaf_variable(self._impl)

            if other._creator_fn is not None:
                mul_fn.add_previous_function(other._creator_fn, other._output_index)
            else:
                mul_fn.add_leaf_variable(other._impl)

            result._creator_fn = mul_fn

        return result

    def _mul_scalar(self, scalar):
        from torch.tensor import Tensor

        result_tensor = self.data * scalar

        if self.volatile:
            return Variable(result_tensor, requires_grad=False, volatile=True)

        result = Variable(result_tensor, requires_grad=self.requires_grad)

        if self.requires_grad:
            mul_fn = _C.MulScalarBackward(scalar)
            mul_fn.num_inputs = 1
            mul_fn.requires_grad = True

            if self._creator_fn is not None:
                mul_fn.add_previous_function(self._creator_fn, self._output_index)
            else:
                mul_fn.add_leaf_variable(self._impl)

            result._creator_fn = mul_fn

        return result

    def __add__(self, other):
        if isinstance(other, Variable):
            return self._add_variable(other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def _add_variable(self, other):
        from torch.tensor import Tensor

        result_tensor = self.data + other.data

        any_volatile = self.volatile or other.volatile
        any_requires_grad = self.requires_grad or other.requires_grad

        if any_volatile:
            return Variable(result_tensor, requires_grad=False, volatile=True)

        result = Variable(result_tensor, requires_grad=any_requires_grad)

        if any_requires_grad:
            add_fn = _C.AddBackward()
            add_fn.num_inputs = 2
            add_fn.requires_grad = True

            if self._creator_fn is not None:
                add_fn.add_previous_function(self._creator_fn, self._output_index)
            else:
                add_fn.add_leaf_variable(self._impl)

            if other._creator_fn is not None:
                add_fn.add_previous_function(other._creator_fn, other._output_index)
            else:
                add_fn.add_leaf_variable(other._impl)

            result._creator_fn = add_fn

        return result

    def __repr__(self):
        return f"Variable(data={self.data}, requires_grad={self.requires_grad})"
