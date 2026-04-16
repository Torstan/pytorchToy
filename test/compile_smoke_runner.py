import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TESTS = [
    ROOT / "test" / "compile.py",
    ROOT / "test" / "compile_partitioned_graph.py",
    ROOT / "test" / "compile_fusion_cpu.py",
    ROOT / "test" / "compile_python_overhead.py",
]

for test_file in TESTS:
    runpy.run_path(str(test_file), run_name="__main__")

print("compile_smoke_runner: ok")
