from torch._functorch.aot_autograd import (
    aot_function,
    aot_module_simplified,
    make_boxed_func,
)
from torch._functorch._aot_autograd.backward_graph import build_backward_graph_or_stub
from torch._functorch._aot_autograd.runtime import attach_compiled_backward
from torch._functorch._aot_autograd.utils import call_signature


assert callable(aot_function)
assert callable(aot_module_simplified)
assert make_boxed_func(lambda x: x)(3) == 3
assert callable(build_backward_graph_or_stub)
assert callable(attach_compiled_backward)
assert callable(call_signature)
