"""
torch.testing -- 测试工具

对应 PyTorch: torch.testing._comparison.assert_close

核心逻辑: |actual - expected| <= atol + rtol * |expected|
"""


def assert_close(actual, expected, *, rtol=1.3e-6, atol=1e-5, **kwargs):
    """
    断言两个张量近似相等。

    判定标准 (逐元素):
        |actual - expected| <= atol + rtol * |expected|

    对应 PyTorch torch.testing.assert_close。
    """
    from torch.tensor import Tensor

    if not isinstance(actual, Tensor) or not isinstance(expected, Tensor):
        raise TypeError(
            f"assert_close expects Tensor inputs, got "
            f"{type(actual).__name__} and {type(expected).__name__}")

    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch: {actual.shape} vs {expected.shape}")

    numel = actual.numel()
    max_err = 0.0
    max_idx = 0
    for i in range(numel):
        a = actual._c.flat_get(i)
        e = expected._c.flat_get(i)
        diff = abs(a - e)
        tol = atol + rtol * abs(e)
        if diff > tol:
            if diff > max_err:
                max_err = diff
                max_idx = i

    if max_err > 0.0:
        act_val = actual._c.flat_get(max_idx)
        exp_val = expected._c.flat_get(max_idx)
        tol_val = atol + rtol * abs(exp_val)
        raise AssertionError(
            f"Tensor values are not close!\n"
            f"  Max absolute difference: {max_err:.6e} "
            f"(up to {tol_val:.6e} allowed)\n"
            f"  At flat index {max_idx}: "
            f"actual={act_val:.6e}, expected={exp_val:.6e}")
