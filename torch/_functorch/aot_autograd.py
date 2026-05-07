"""
Public AOTAutograd facade.
"""

from torch._functorch._aot_autograd import (
    AOTFunction,
    aot_function,
    aot_module_simplified,
    make_boxed_func,
)

__all__ = [
    "AOTFunction",
    "aot_function",
    "aot_module_simplified",
    "make_boxed_func",
]
