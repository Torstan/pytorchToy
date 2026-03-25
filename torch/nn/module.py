"""
nn.Module — 神经网络模块基类

参考 PyTorch nn.Module 的核心机制:
- _parameters / _modules / _buffers 字典管理
- __setattr__ / __getattr__ 属性拦截
- __call__ → forward()
- parameters() 递归遍历
"""

from torch.nn.parameter import Parameter


class Module:
    """所有神经网络模块的基类"""

    def __init__(self):
        # 通过 object.__setattr__ 绕过自身的 __setattr__
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, 'training', True)

    def __setattr__(self, name, value):
        """
        属性拦截: 自动分发到对应的字典
        - Parameter → _parameters
        - Module → _modules
        - 其他 → __dict__
        """
        params = self.__dict__.get('_parameters')
        modules = self.__dict__.get('_modules')
        buffers = self.__dict__.get('_buffers')

        if isinstance(value, Parameter):
            if params is not None:
                # 互斥清理
                if modules is not None and name in modules:
                    del modules[name]
                if buffers is not None and name in buffers:
                    del buffers[name]
                params[name] = value
                return
        elif isinstance(value, Module):
            if modules is not None:
                if params is not None and name in params:
                    del params[name]
                if buffers is not None and name in buffers:
                    del buffers[name]
                modules[name] = value
                return

        # 普通属性
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        当 __dict__ 查找失败时触发，按顺序搜索:
        _parameters → _buffers → _modules
        """
        _parameters = self.__dict__.get('_parameters', {})
        if name in _parameters:
            return _parameters[name]

        _buffers = self.__dict__.get('_buffers', {})
        if name in _buffers:
            return _buffers[name]

        _modules = self.__dict__.get('_modules', {})
        if name in _modules:
            return _modules[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        """调用 forward()"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """子类必须实现"""
        raise NotImplementedError

    def parameters(self):
        """递归返回所有参数 (DFS)"""
        for param in self._parameters.values():
            if param is not None:
                yield param
        for module in self._modules.values():
            if module is not None:
                yield from module.parameters()

    def named_parameters(self, prefix=''):
        """递归返回所有 (name, parameter) 对"""
        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param
        for name, module in self._modules.items():
            if module is not None:
                sub_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_parameters(sub_prefix)

    def register_buffer(self, name, tensor):
        """注册 buffer (非参数状态)"""
        self._buffers[name] = tensor

    def train(self, mode=True):
        """设置训练模式"""
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self):
        """设置评估模式"""
        return self.train(False)

    def zero_grad(self):
        """清零所有参数的梯度"""
        for param in self.parameters():
            param.grad = None

    def state_dict(self):
        """返回模型状态字典"""
        result = {}
        for name, param in self.named_parameters():
            result[name] = param
        return result

    def __repr__(self):
        lines = [f"{type(self).__name__}("]
        for name, module in self._modules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return '\n'.join(lines)
