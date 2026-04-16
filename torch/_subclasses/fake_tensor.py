"""
最小 FakeTensor 实现。

这里只保留 meta 信息，不持有真实 storage。
"""

from torch.tensor import Tensor, float32


def _contiguous_stride(shape):
    dims = list(shape)
    if not dims:
        return ()
    stride = [0] * len(dims)
    running = 1
    for index in range(len(dims) - 1, -1, -1):
        stride[index] = running
        running *= dims[index]
    return tuple(stride)


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype=None,
        requires_grad=False,
        stride=None,
        device="cpu",
    ):
        self.shape = tuple(shape)
        self.dtype = dtype if dtype is not None else float32
        self.requires_grad = requires_grad
        self.device = device
        self._stride = tuple(stride) if stride is not None else _contiguous_stride(self.shape)

    @classmethod
    def from_tensor(cls, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"expected Tensor, got {type(tensor)}")
        return cls(
            tensor.shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            stride=tensor.stride(),
            device=tensor.device,
        )

    def clone(self):
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            stride=self._stride,
            device=self.device,
        )

    def dim(self):
        return len(self.shape)

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def size(self, dim=None):
        if dim is None:
            return list(self.shape)
        return self.shape[dim]

    def stride(self):
        return self._stride

    def is_contiguous(self):
        return self._stride == _contiguous_stride(self.shape)

    def contiguous(self):
        return FakeTensor(
            self.shape,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return FakeTensor(
            tuple(shape),
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            device=self.device,
        )

    def reshape(self, *shape):
        return self.view(*shape)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return FakeTensor((), dtype=self.dtype, requires_grad=self.requires_grad, device=self.device)
        dims = list(self.shape)
        if dim < 0:
            dim = len(dims) + dim
        if keepdim:
            dims[dim] = 1
        else:
            dims.pop(dim)
        return FakeTensor(tuple(dims), dtype=self.dtype, requires_grad=self.requires_grad, device=self.device)

    def __repr__(self):
        return (
            "FakeTensor("
            f"shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad}"
            ")"
        )


class FakeTensorMode:
    def __init__(self, allow_non_fake_inputs=True):
        self.allow_non_fake_inputs = allow_non_fake_inputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def from_tensor(self, tensor):
        return FakeTensor.from_tensor(tensor)
