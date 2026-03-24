// ============================================================
// 自动生成的文件 — 请勿手动修改!
// 由 codegen.py 从 native_functions.yaml 生成
// ============================================================
// Tensor 类的算子方法绑定
        .def("add", &Tensor::add)
        .def("mul", &Tensor::mul)
        .def("matmul", &Tensor::matmul)
        .def("relu", &Tensor::relu)
        .def("sum", &Tensor::sum)
        .def("contiguous", &Tensor::contiguous)
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("slice", &Tensor::slice, py::arg("dim"), py::arg("start"), py::arg("end"))
        .def("reshape", &Tensor::reshape, py::arg("shape"))
        .def("expand", &Tensor::expand, py::arg("sizes"))

// Tensor 运算符重载绑定
        .def("__add__", [](const Tensor& a, const Tensor& b) { return native::add(a, b); })
        .def("__mul__", [](const Tensor& a, const Tensor& b) { return native::mul(a, b); })
