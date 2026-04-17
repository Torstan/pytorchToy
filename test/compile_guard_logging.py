import contextlib
import io

import torch


def fn(x, scale):
    return torch.sin(x) * scale


torch._logging.set_logs(guards=True, recompiles=True)

compiled = torch.compile(fn, backend="eager")

x1 = torch.randn(2, 3)
x2 = torch.randn(4, 3)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    out1 = compiled(x1, 2.0)
    out2 = compiled(x2, 2.0)

log_output = buf.getvalue()

assert "=== GUARDS ===" in log_output, log_output
assert "arg0: tensor shape=(2, 3)" in log_output, log_output
assert "arg1: float value=2.0" in log_output, log_output
assert "=== RECOMPILE ===" in log_output, log_output
assert "cached_variants=1" in log_output, log_output
assert "arg0: tensor shape=(4, 3)" in log_output, log_output

torch.testing.assert_close(out1, fn(x1, 2.0))
torch.testing.assert_close(out2, fn(x2, 2.0))

torch._logging.set_logs(guards=False, recompiles=False)

print("compile_guard_logging: ok")
