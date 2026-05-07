"""
Shared operator helpers for the toy compile stack.
"""

UNARY_POINTWISE_TARGETS = frozenset({"sin", "cos", "relu", "tanh", "neg"})
BINARY_POINTWISE_TARGETS = frozenset({"add", "sub", "mul", "div"})
POINTWISE_TARGETS = UNARY_POINTWISE_TARGETS | BINARY_POINTWISE_TARGETS

TORCH_OPERATOR_NAMES = frozenset({
    "sin",
    "cos",
    "exp",
    "log",
    "relu",
    "tanh",
    "sum",
    "view",
    "reshape",
    "mm",
    "addmm",
})

TENSOR_METHOD_NAMES = frozenset({
    "sin",
    "cos",
    "exp",
    "log",
    "relu",
    "tanh",
    "sum",
    "view",
    "reshape",
    "mm",
    "t",
    "gt",
})


def target_name(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        return target.__name__
    return repr(target)


def normalize_shape_args(args):
    if len(args) == 2 and isinstance(args[1], (tuple, list)):
        return tuple(args[1])
    return tuple(args[1:])


def broadcast_shapes(lhs_shape, rhs_shape, *, error_type=RuntimeError):
    lhs = list(lhs_shape)
    rhs = list(rhs_shape)
    result = []
    while lhs or rhs:
        left = lhs.pop() if lhs else 1
        right = rhs.pop() if rhs else 1
        if left == 1:
            result.append(right)
            continue
        if right == 1 or left == right:
            result.append(left)
            continue
        raise error_type(f"cannot broadcast shapes {lhs_shape} and {rhs_shape}")
    result.reverse()
    return tuple(result)


EAGER_OP_TABLE = {}


def register_eager_op(name):
    def decorator(fn):
        EAGER_OP_TABLE[name] = fn
        return fn
    return decorator


@register_eager_op("sin")
def _op_sin(args, kwargs):
    del kwargs
    return args[0].sin()


@register_eager_op("cos")
def _op_cos(args, kwargs):
    del kwargs
    return args[0].cos()


@register_eager_op("exp")
def _op_exp(args, kwargs):
    del kwargs
    return args[0].exp()


@register_eager_op("log")
def _op_log(args, kwargs):
    del kwargs
    return args[0].log()


@register_eager_op("add")
def _op_add(args, kwargs):
    del kwargs
    return args[0] + args[1]


@register_eager_op("sub")
def _op_sub(args, kwargs):
    del kwargs
    return args[0] - args[1]


@register_eager_op("mul")
def _op_mul(args, kwargs):
    del kwargs
    return args[0] * args[1]


@register_eager_op("div")
def _op_div(args, kwargs):
    del kwargs
    return args[0] / args[1]


@register_eager_op("neg")
def _op_neg(args, kwargs):
    del kwargs
    return -args[0]


@register_eager_op("relu")
def _op_relu(args, kwargs):
    del kwargs
    return args[0].relu()


@register_eager_op("tanh")
def _op_tanh(args, kwargs):
    del kwargs
    return args[0].tanh()


@register_eager_op("sum")
def _op_sum(args, kwargs):
    return args[0].sum(**kwargs)


@register_eager_op("gt")
def _op_gt(args, kwargs):
    del kwargs
    return args[0].gt(args[1])


@register_eager_op("t")
def _op_t(args, kwargs):
    del kwargs
    return args[0].t()


@register_eager_op("mm")
def _op_mm(args, kwargs):
    del kwargs
    return args[0].mm(args[1])


@register_eager_op("view")
def _op_view(args, kwargs):
    del kwargs
    return args[0].view(*normalize_shape_args(args))


@register_eager_op("reshape")
def _op_reshape(args, kwargs):
    del kwargs
    return args[0].reshape(*normalize_shape_args(args))


@register_eager_op("addmm")
def _op_addmm(args, kwargs):
    del kwargs
    bias, lhs, rhs = args
    return lhs.mm(rhs) + bias


@register_eager_op("layer_norm")
def _op_layer_norm(args, kwargs):
    import torch.nn.functional as F

    return F.layer_norm(*args, **kwargs)


@register_eager_op("call_callable")
def _op_call_callable(args, kwargs):
    return args[0](*args[1:], **kwargs)


def run_eager_target(target, args, kwargs):
    op_fn = EAGER_OP_TABLE.get(target)
    if op_fn is not None:
        return op_fn(args, kwargs)
    if not isinstance(target, str):
        return target(*args, **kwargs)
    raise RuntimeError(f"unsupported compiled target: {target}")
