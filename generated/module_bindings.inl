// ============================================================
// 自动生成的文件 — 请勿手动修改!
// 由 codegen.py 从 native_functions.yaml 生成
// ============================================================
// 模块级算子函数绑定
    m.def("add", &native::add, py::arg("a"), py::arg("b"));
    m.def("mul", &native::mul, py::arg("a"), py::arg("b"));
    m.def("matmul", &native::matmul, py::arg("a"), py::arg("b"));
    m.def("relu", &native::relu, py::arg("input"));
    m.def("sum", &native::sum, py::arg("input"));
    m.def("contiguous", &native::contiguous, py::arg("self"));
    m.def("transpose", &native::transpose, py::arg("self"), py::arg("dim0"), py::arg("dim1"));
    m.def("slice", &native::slice, py::arg("self"), py::arg("dim"), py::arg("start"), py::arg("end"));
    m.def("reshape", &native::reshape, py::arg("self"), py::arg("shape"));
    m.def("expand", &native::expand, py::arg("self"), py::arg("sizes"));
