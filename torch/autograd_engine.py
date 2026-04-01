"""
Autograd Engine — C++ 主引擎的 Python 辅助层
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _C


class no_grad:
    """Context manager: 禁用梯度记录"""

    def __enter__(self):
        self._prev = _C.is_grad_enabled()
        _C.set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        _C.set_grad_enabled(self._prev)
