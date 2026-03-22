// ============================================================
// Python ↔ C++ 绑定层
// 对应 PyTorch: torch/csrc/ 目录下的 pybind11 绑定
//
// PyTorch 调用链:
//   Python: torch.add(a, b)
//     → torch/_C/ (pybind11 绑定，类似本文件)
//       → c10::Dispatcher (算子分派)
//         → at::native::add() (C++ kernel)
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "Mini PyTorch: 模拟 TensorImpl → TensorBase → Tensor 三层结构";

    // ============================================================
    // 绑定 TensorImpl（通常不直接暴露给 Python，这里仅供演示）
    // 对应 PyTorch: c10/core/TensorImpl.h
    // ============================================================
    py::class_<TensorImpl>(m, "TensorImpl")
        .def("dim", &TensorImpl::dim)
        .def("numel", &TensorImpl::numel)
        .def_readonly("sizes", &TensorImpl::sizes_)
        .def_readonly("strides", &TensorImpl::strides_)
        .def_readwrite("requires_grad", &TensorImpl::requires_grad_);

    // ============================================================
    // 绑定 TensorBase — 不包含算子方法
    // 对应 PyTorch: aten/src/ATen/core/TensorBase.h
    // ============================================================
    py::class_<TensorBase>(m, "TensorBase")
        .def("dim", &TensorBase::dim)
        .def("numel", &TensorBase::numel)
        .def("sizes", &TensorBase::sizes)
        .def("strides", &TensorBase::strides)
        .def("requires_grad", &TensorBase::requires_grad)
        .def("defined", &TensorBase::defined)
        .def("use_count", &TensorBase::use_count)
        .def("__repr__", &TensorBase::repr);

    // ============================================================
    // 绑定 Tensor — 继承 TensorBase，增加算子方法
    // 对应 PyTorch: aten/src/ATen/templates/TensorBody.h
    //
    // 这里体现了关键设计：
    // - TensorBase 只有元信息方法，不依赖 native_functions.yaml
    // - Tensor 继承 TensorBase，增加由代码生成的算子方法
    // - 修改算子时只需重编译 Tensor 相关代码
    // ============================================================
    py::class_<Tensor, TensorBase>(m, "Tensor")
        .def(py::init([](std::vector<int> shape, float fill) {
            return Tensor(IntrusivePtr(new TensorImpl(shape, fill)));
        }), py::arg("shape"), py::arg("fill_value") = 0.0f)
        // 数据访问
        .def("__getitem__", [](const Tensor& t, int i) { return t.data_ptr()[i]; })
        .def("__setitem__", [](Tensor& t, int i, float v) { t.data_ptr()[i] = v; })
        // 算子方法 + 运算符重载（由 codegen.py 从 native_functions.yaml 自动生成）
#include "generated/tensor_bindings.inl"
        // 演示 TensorImpl 的共享（多个 Tensor 可以指向同一个 TensorImpl）
        .def("data_ptr_id", [](const Tensor& t) {
            return reinterpret_cast<uintptr_t>(t.unsafeGetTensorImpl());
        });

    // ============================================================
    // 模块级工厂函数
    // 对应 PyTorch: torch.ones(), torch.empty() 等
    // ============================================================
    m.def("ones", &native::ones, py::arg("shape"));
    m.def("fill", &native::fill, py::arg("shape"), py::arg("value"));
    m.def("empty", &native::empty, py::arg("shape"));

    // 模块级算子函数（由 codegen.py 从 native_functions.yaml 自动生成）
#include "generated/module_bindings.inl"
}
