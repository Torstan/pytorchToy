import torch


backends = torch._dynamo.list_backends()

assert "default" in backends
assert "eager" in backends
assert "inductor" in backends

print("dynamo_backend_registry: ok")
