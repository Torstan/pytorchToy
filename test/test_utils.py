import importlib


def eager_compiler(gm, example_inputs):
    del example_inputs

    def compiled(*args):
        return gm(*args)

    return compiled


def make_compiler(name, seen):
    def compiler(gm, example_inputs):
        del example_inputs
        seen[name] += 1

        def compiled(*args):
            return gm(*args)

        return compiled

    return compiler


def make_counting_compile_graph_module(counts):
    """Returns (wrapped_fn, original_fn) for monkey-patching compile_graph_module."""
    compile_fx_mod = importlib.import_module("torch._inductor.compile_fx")
    original = compile_fx_mod.compile_graph_module

    def wrapped(gm, example_inputs, **kwargs):
        kernel = original(gm, example_inputs, **kwargs)
        counts["compile"] += 1

        class WrappedKernel:
            def run(self, args):
                counts["run"] += 1
                return kernel.run(args)

        return WrappedKernel()

    return wrapped, original
