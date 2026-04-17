import importlib
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TESTS = [
    ROOT / "test" / "compile.py",
    ROOT / "test" / "compile_graph_break_inductor_early_return.py",
    ROOT / "test" / "compile_graph_break_inductor_loop_carry.py",
    ROOT / "test" / "compile_graph_break_inductor_mixed_early_return.py",
    ROOT / "test" / "compile_graph_break_inductor_resume.py",
    ROOT / "test" / "compile_partitioned_graph.py",
    ROOT / "test" / "compile_fusion_cpu.py",
    ROOT / "test" / "compile_python_overhead.py",
]

compile_fx_mod = importlib.import_module("torch._inductor.compile_fx")
_true_compile_fx = compile_fx_mod.compile_fx

for test_file in TESTS:
    try:
        runpy.run_path(str(test_file), run_name="__main__")
    finally:
        compile_fx_mod.compile_fx = _true_compile_fx

print("compile_smoke_runner: ok")
