CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -fPIC -funroll-loops

PYBIND11_INCLUDE := $(shell python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

TARGET := _C$(EXT_SUFFIX)
NN_TARGET := _nn_C$(EXT_SUFFIX)

# 生成的文件（由 codegen.py 从 native_functions.yaml 生成）
GENERATED := generated/tensor_methods.h generated/dispatch.h generated/tensor_bindings.inl generated/module_bindings.inl

.PHONY: all clean run codegen demo_codegen

all: $(TARGET) $(NN_TARGET)

# ============================================================
# 代码生成: native_functions.yaml → generated/*.h
# 对应 PyTorch 的 torchgen 步骤
# ============================================================
$(GENERATED): native_functions.yaml codegen.py
	python3 codegen.py

codegen: $(GENERATED)

# ============================================================
# 分离编译，体现 TensorBase 不依赖算子定义
#
# tensor_base.o:
#   - 依赖: tensor_impl.h, tensor.h
#   - 不依赖: native_functions.yaml, ops.h, generated/*
#   → 修改算子时不需要重新编译！
#
# bindings.o:
#   - 依赖: 所有头文件 + generated/*
#   → 修改算子时需要重新编译
# ============================================================

tensor_base.o: tensor_base.cpp tensor_base.h tensor_impl.h
	@echo ">>> 编译 tensor_base.o (不依赖算子定义)"
	$(CXX) $(CXXFLAGS) -I$(PYTHON_INCLUDE) -I. -c tensor_base.cpp -o $@

AUTOGRAD_HEADERS := autograd/variable.h autograd/function.h autograd/grad_ops.h autograd/nn_grad_ops.h autograd/engine.h autograd/py_function.h autograd/autograd_ops.h

bindings.o: bindings.cpp ops.h tensor.h tensor_base.h tensor_impl.h $(GENERATED) $(AUTOGRAD_HEADERS) $(NN_HEADERS)
	@echo ">>> 编译 bindings.o (依赖算子定义 + autograd)"
	$(CXX) $(CXXFLAGS) -I$(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE) -I. -c bindings.cpp -o $@

$(TARGET): tensor_base.o bindings.o
	@echo ">>> 链接 $(TARGET)"
	$(CXX) -shared tensor_base.o bindings.o -o $@

# ============================================================
# nn 模块编译
# ============================================================
NN_HEADERS := nn/ops.h util/math.h util/tensor_ops.h

nn_bindings.o: nn/nn_bindings.cpp $(NN_HEADERS) ops.h tensor.h tensor_base.h tensor_impl.h
	@echo ">>> 编译 nn_bindings.o (nn C++ 内核)"
	$(CXX) $(CXXFLAGS) -I$(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE) -I. -c nn/nn_bindings.cpp -o $@

$(NN_TARGET): nn_bindings.o
	@echo ">>> 链接 $(NN_TARGET)"
	$(CXX) -shared nn_bindings.o -o $@

run: $(TARGET) $(NN_TARGET)
	python3 demo.py

# ============================================================
# 演示: 修改算子不需要重新编译 TensorBase
#
# 运行 make demo_codegen 观察:
# 1. 首次构建: tensor_base.o 和 bindings.o 都会编译
# 2. 修改 native_functions.yaml 后再次构建:
#    - codegen.py 重新生成代码
#    - bindings.o 重新编译 (依赖 generated/*)
#    - tensor_base.o 不会重新编译! (不依赖 generated/*)
# ============================================================
demo_codegen:
	@echo ""
	@echo "=============================="
	@echo "步骤 1: 首次完整构建"
	@echo "=============================="
	$(MAKE) clean
	$(MAKE) all
	@echo ""
	@echo "=============================="
	@echo "步骤 2: 模拟修改 native_functions.yaml (touch)"
	@echo "=============================="
	touch native_functions.yaml
	@echo ""
	@echo "=============================="
	@echo "步骤 3: 增量构建 — 观察 tensor_base.o 不会重编译"
	@echo "=============================="
	$(MAKE) all
	@echo ""
	@echo "=============================="
	@echo "结论: 修改算子只重编译 bindings.o，tensor_base.o 不受影响"
	@echo "=============================="

clean:
	rm -f $(TARGET) *.o *.so
	rm -rf generated/
	rm -rf build *.egg-info
