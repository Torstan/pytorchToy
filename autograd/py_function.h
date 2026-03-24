#pragma once

#include "function.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// ============================================================
// PyFunction — 桥接 Python 自定义 Function 的 backward 回调
//
// 当 C++ Engine 执行到此节点时，调用 Python 对象的
// _do_backward() 方法，该方法负责：
// 1. 将 _C.Tensor 包装为 Python Tensor
// 2. 调用用户定义的 backward()
// 3. 将结果拆包为 _C.Tensor 返回
// ============================================================

class __attribute__((visibility("hidden"))) PyFunction : public AutogradFunction {
public:
    py::object py_func;  // Python Function 实例的引用

    explicit PyFunction(py::object func) : py_func(std::move(func)) {}

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        // 将 C++ Tensor 传给 Python
        py::list py_grads;
        for (const auto& g : grad_outputs) {
            py_grads.append(py::cast(g));
        }

        // 调用 Python 端的 _do_backward 方法
        py::object result = py_func.attr("_do_backward")(py_grads);

        // 将 Python 返回值转回 C++ Tensor
        std::vector<Tensor> out;
        if (py::isinstance<py::tuple>(result)) {
            for (auto item : py::cast<py::tuple>(result)) {
                out.push_back(py::cast<Tensor>(item));
            }
        } else if (py::isinstance<py::list>(result)) {
            for (auto item : py::cast<py::list>(result)) {
                out.push_back(py::cast<Tensor>(item));
            }
        } else {
            out.push_back(py::cast<Tensor>(result));
        }
        return out;
    }
};
