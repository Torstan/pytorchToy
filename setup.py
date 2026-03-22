# 构建脚本
# 对应 PyTorch 的 setup.py，使用 pybind11 编译 C++ 扩展
# PyTorch 实际使用 CMake + setuptools，这里简化为纯 setuptools

from setuptools import setup, Extension
import pybind11

ext = Extension(
    "_C",  # 模块名，模拟 torch._C
    sources=["bindings.cpp"],
    include_dirs=[pybind11.get_include(), "."],
    language="c++",
    extra_compile_args=["-std=c++17", "-O2"],
)

setup(
    name="mini_torch",
    version="0.1",
    ext_modules=[ext],
)
