import time
import statistics
import torch

torch.set_num_threads(1)

def median_time(fn, *args, warmup=10, repeat=30):
    for _ in range(warmup):
        fn(*args)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def fuse_demo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    c = a + b
    d = torch.relu(c * 2.0)
    return d

compiled_fuse_demo = torch.compile(fuse_demo, backend="inductor", fullgraph=True)

x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)

ref = fuse_demo(x, y)
out = compiled_fuse_demo(x, y)
torch.testing.assert_close(out, ref)

eager_t = median_time(fuse_demo, x, y)
compiled_t = median_time(compiled_fuse_demo, x, y)

print(f"eager   : {eager_t:.6f}s")
print(f"compile : {compiled_t:.6f}s")
print(f"speedup : {eager_t / compiled_t:.2f}x")

