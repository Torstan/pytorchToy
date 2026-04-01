import time
import torch

torch.set_num_threads(1)

def avg_time_per_call(fn, x, calls=2000, warmup=20):
    for _ in range(warmup):
        fn(x)
    t0 = time.perf_counter()
    for _ in range(calls):
        fn(x)
    return (time.perf_counter() - t0) / calls

def python_overhead_demo(x):
    for _ in range(100):
        x = x + 1.01
        x = x * 0.99
        x = torch.relu(x)
    return x

compiled_demo = torch.compile(python_overhead_demo, backend="inductor", fullgraph=True)

x = torch.randn(128)

ref = python_overhead_demo(x)
out = compiled_demo(x)
torch.testing.assert_close(out, ref)

eager_t = avg_time_per_call(python_overhead_demo, x)
compiled_t = avg_time_per_call(compiled_demo, x)

print(f"eager avg/call   : {eager_t * 1e6:.2f} us")
print(f"compile avg/call : {compiled_t * 1e6:.2f} us")
print(f"speedup          : {eager_t / compiled_t:.2f}x")

