from torch._compile.ops import (
    BINARY_POINTWISE_TARGETS,
    TENSOR_METHOD_NAMES,
    TORCH_OPERATOR_NAMES,
    UNARY_POINTWISE_TARGETS,
    broadcast_shapes,
    normalize_shape_args,
    target_name,
)


def sample():
    pass


assert target_name("add") == "add"
assert target_name(sample) == "sample"
assert normalize_shape_args(("x", (2, 3))) == (2, 3)
assert normalize_shape_args(("x", 2, 3)) == (2, 3)
assert broadcast_shapes((2, 1), (1, 3)) == (2, 3)
assert "sin" in UNARY_POINTWISE_TARGETS
assert "add" in BINARY_POINTWISE_TARGETS
assert "relu" in TORCH_OPERATOR_NAMES
assert "view" in TENSOR_METHOD_NAMES
