"""
最小 PrimTorch primitives。
"""


def sin(x):
    return x.sin()


def cos(x):
    return x.cos()


def relu(x):
    return x.relu()


def tanh(x):
    return x.tanh()


def neg(x):
    return -x


def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


def div(x, y):
    return x / y


def sum(x, dim=None, keepdim=False):
    return x.sum(dim=dim, keepdim=keepdim)


def view(x, shape):
    return x.view(shape)


def reshape(x, shape):
    return x.reshape(shape)


def mm(x, y):
    return x.mm(y)
