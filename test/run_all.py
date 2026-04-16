#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "test"
DEFAULT_SKIP_DIRS = {"__pycache__", "perf"}


def _discover_python_tests(include_perf):
    tests = []
    skip_dirs = {"__pycache__"} if include_perf else DEFAULT_SKIP_DIRS
    for path in sorted(TEST_DIR.rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        tests.append(path)
    return tests


def _discover_cpp_tests(include_perf):
    tests = []
    for path in sorted(TEST_DIR.rglob("*.cpp")):
        rel_parts = path.relative_to(TEST_DIR).parts
        if not include_perf and rel_parts and rel_parts[0] == "perf":
            continue
        tests.append(path)
    return tests


def _run_command(cmd, cwd, timeout=None):
    start = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        timeout=timeout,
        text=True,
        capture_output=True,
    )
    duration = time.time() - start
    return completed, duration


def _run_python_test(path, timeout):
    return _run_command([sys.executable, str(path)], ROOT, timeout=timeout)


def _run_cpp_test(path, timeout):
    rel_no_suffix = path.relative_to(ROOT).with_suffix("")
    build_cmd = ["make", str(rel_no_suffix)]
    build, build_duration = _run_command(build_cmd, ROOT, timeout=timeout)
    if build.returncode != 0:
        return {
            "name": str(path.relative_to(ROOT)),
            "kind": "cpp",
            "ok": False,
            "duration": build_duration,
            "stage": "build",
            "stdout": build.stdout,
            "stderr": build.stderr,
        }

    binary_path = ROOT / rel_no_suffix
    run, run_duration = _run_command([str(binary_path)], ROOT, timeout=timeout)
    return {
        "name": str(path.relative_to(ROOT)),
        "kind": "cpp",
        "ok": run.returncode == 0,
        "duration": build_duration + run_duration,
        "stage": "run",
        "stdout": run.stdout,
        "stderr": run.stderr,
    }


def _run_with_timeout_guard(runner, path, timeout):
    try:
        completed, duration = runner(path, timeout)
        return {
            "name": str(path.relative_to(ROOT)),
            "kind": "python",
            "ok": completed.returncode == 0,
            "duration": duration,
            "stage": "run",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": str(path.relative_to(ROOT)),
            "kind": "python",
            "ok": False,
            "duration": timeout if timeout is not None else 0.0,
            "stage": "timeout",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def _run_cpp_with_timeout_guard(path, timeout):
    try:
        return _run_cpp_test(path, timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "name": str(path.relative_to(ROOT)),
            "kind": "cpp",
            "ok": False,
            "duration": timeout if timeout is not None else 0.0,
            "stage": "timeout",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def main():
    parser = argparse.ArgumentParser(description="Run all tests under test/.")
    parser.add_argument("--python-only", action="store_true", help="Only run Python tests.")
    parser.add_argument("--cpp-only", action="store_true", help="Only run C++ tests.")
    parser.add_argument(
        "--include-perf",
        action="store_true",
        help="Also include tests under test/perf/.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-test timeout in seconds. Default: 300.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list discovered tests.",
    )
    args = parser.parse_args()

    if args.python_only and args.cpp_only:
        raise SystemExit("--python-only and --cpp-only cannot be used together")

    python_tests = [] if args.cpp_only else _discover_python_tests(args.include_perf)
    cpp_tests = [] if args.python_only else _discover_cpp_tests(args.include_perf)

    discovered = [str(path.relative_to(ROOT)) for path in python_tests + cpp_tests]
    if args.list:
        for item in discovered:
            print(item)
        print(f"\nTotal: {len(discovered)}")
        return 0

    results = []

    for path in python_tests:
        rel = path.relative_to(ROOT)
        print(f"[PY]  {rel}")
        result = _run_with_timeout_guard(_run_python_test, path, args.timeout)
        results.append(result)
        status = "PASS" if result["ok"] else "FAIL"
        print(f"      {status} {result['duration']:.2f}s")

    for path in cpp_tests:
        rel = path.relative_to(ROOT)
        print(f"[CPP] {rel}")
        result = _run_cpp_with_timeout_guard(path, args.timeout)
        results.append(result)
        status = "PASS" if result["ok"] else "FAIL"
        print(f"      {status} {result['duration']:.2f}s")

    failures = [result for result in results if not result["ok"]]
    print()
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Total : {len(results)}")

    if failures:
        print("\nFailures:")
        for result in failures:
            print(f"- {result['name']} ({result['stage']})")
            if result["stdout"]:
                print("  stdout:")
                for line in result["stdout"].splitlines():
                    print(f"    {line}")
            if result["stderr"]:
                print("  stderr:")
                for line in result["stderr"].splitlines():
                    print(f"    {line}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
