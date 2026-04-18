import torch


compiled = {"count": 0, "gm": None, "inputs": None}


class ExpModule(torch.nn.Module):
    def forward(self, x):
        return torch.exp(2 * x)


def fn(x, mod):
    return torch.log(mod(x))


def backend(gm, example_inputs):
    compiled["count"] += 1
    compiled["gm"] = gm
    compiled["inputs"] = list(example_inputs)

    def compiled_fn(*args):
        return gm(*args)

    return compiled_fn


opt = torch.compile(fn, backend=backend)

x = torch.randn(3, 3)
mod = ExpModule()

torch.testing.assert_close(opt(x, mod), fn(x, mod))

assert compiled["count"] == 1
assert compiled["gm"] is not None
assert len(compiled["inputs"]) == 1, compiled["inputs"]
assert compiled["inputs"][0].shape == (3, 3)

readable = compiled["gm"].print_readable(print_output=False)
assert "call_callable" not in readable, readable
assert "torch.exp" in readable, readable
assert "torch.log" in readable, readable
assert "def compiled_graph(x):" in readable, readable

print("compile_module_arg_inline: ok")
