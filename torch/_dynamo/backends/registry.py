"""
Minimal Dynamo backend registry.
"""

_BACKENDS = {}


def register_backend(name):
    def decorator(fn):
        _BACKENDS[name] = fn
        return fn
    return decorator


def lookup_backend(backend):
    if callable(backend) and not isinstance(backend, str):
        return backend
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend}. Available: {list(_BACKENDS.keys())}"
            )
        return _BACKENDS[backend]
    raise TypeError(f"backend must be str or callable, got {type(backend)}")


def list_backends():
    return sorted(_BACKENDS.keys())


@register_backend("eager")
def eager_backend(graph_module, example_inputs):
    del example_inputs

    def compiled_fn(*args):
        return graph_module(*args)

    return compiled_fn


@register_backend("inductor")
def inductor_backend(graph_module, example_inputs):
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(graph_module, example_inputs)


@register_backend("default")
def default_backend(graph_module, example_inputs):
    del example_inputs

    def compiled_fn(*args):
        return graph_module(*args)

    return compiled_fn
