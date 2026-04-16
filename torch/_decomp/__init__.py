"""
最小 decomposition registry。
"""


decomposition_table = {}


def register_decomposition(target):
    def decorator(fn):
        decomposition_table[target] = fn
        return fn

    return decorator


def get_decompositions(targets=None):
    if targets is None:
        return dict(decomposition_table)
    return {target: decomposition_table[target] for target in targets if target in decomposition_table}
