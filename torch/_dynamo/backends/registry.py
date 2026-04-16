"""
最小 backend registry。

当前直接复用 torch._compile.backend 的注册表。
"""

from torch._compile.backend import list_backends as _list_backends
from torch._compile.backend import lookup_backend as _lookup_backend


def lookup_backend(backend):
    return _lookup_backend(backend)


def list_backends():
    return _list_backends()
