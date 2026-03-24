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

// 从嵌套 Python list 推断 shape 并提取扁平数据
// 例如 [[1,2],[3,4]] → shape={2,2}, data={1,2,3,4}
static void infer_shape_and_flatten(py::handle obj,
                                     std::vector<int>& shape,
                                     std::vector<float>& data,
                                     int depth) {
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        auto seq = py::cast<py::list>(obj);
        int len = static_cast<int>(seq.size());
        if (static_cast<int>(shape.size()) <= depth)
            shape.push_back(len);
        else if (shape[depth] != len)
            throw std::runtime_error("Jagged nested list: inconsistent sizes");
        for (int i = 0; i < len; i++)
            infer_shape_and_flatten(seq[i], shape, data, depth + 1);
    } else {
        data.push_back(py::cast<float>(obj));
    }
}

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
        .def("is_contiguous", &TensorBase::is_contiguous)
        .def("storage_offset", &TensorBase::storage_offset)
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
        // 构造方式1: shape + 填充值，如 Tensor([2,3], 1.0)
        .def(py::init([](std::vector<int> shape, float fill) {
            return Tensor(TensorImplPtr(new TensorImpl(shape, fill)));
        }), py::arg("shape"), py::arg("fill_value") = 0.0f)
        // 构造方式2: 从嵌套 list 创建，如 Tensor([[[1,2],[3,4]]])
        // 模拟 torch.tensor(data)
        .def_static("from_data", [](py::list data) {
            std::vector<int> shape;
            std::vector<float> flat;
            infer_shape_and_flatten(data, shape, flat, 0);
            auto* impl = new TensorImpl(shape, 0.0f);
            std::copy(flat.begin(), flat.end(), impl->data_ptr());
            return Tensor(TensorImplPtr(impl));
        }, py::arg("data"))
        // 数据访问 — 单整数索引: t[i] 返回子 tensor (view)
        // 对 1D tensor 返回 0-dim tensor（标量 tensor），与 PyTorch 行为一致
        .def("__getitem__", [](const Tensor& t, int i) {
            auto* impl = t.unsafeGetTensorImpl();
            if (i < 0) i += impl->sizes_[0];
            int new_offset = impl->storage_offset_ + i * impl->strides_[0];
            std::vector<int> new_sizes(impl->sizes_.begin() + 1, impl->sizes_.end());
            std::vector<int> new_strides(impl->strides_.begin() + 1, impl->strides_.end());
            auto* vi = new TensorImpl(impl->storage_, new_offset,
                                       std::move(new_sizes), std::move(new_strides));
            return Tensor(TensorImplPtr(vi));
        })
        // 多维索引: t[i, j, ...] — 逐维度降维，返回子 tensor view
        .def("__getitem__", [](const Tensor& t, py::tuple idx) {
            auto* impl = t.unsafeGetTensorImpl();
            int nidx = static_cast<int>(py::len(idx));
            if (nidx > impl->dim())
                throw std::runtime_error("too many indices for tensor");
            int new_offset = impl->storage_offset_;
            for (int d = 0; d < nidx; d++) {
                int i = py::cast<int>(idx[d]);
                if (i < 0) i += impl->sizes_[d];
                new_offset += i * impl->strides_[d];
            }
            std::vector<int> new_sizes(impl->sizes_.begin() + nidx, impl->sizes_.end());
            std::vector<int> new_strides(impl->strides_.begin() + nidx, impl->strides_.end());
            auto* vi = new TensorImpl(impl->storage_, new_offset,
                                       std::move(new_sizes), std::move(new_strides));
            return Tensor(TensorImplPtr(vi));
        })
        .def("__setitem__", [](Tensor& t, int i, float v) { t.data_ptr()[i] = v; })
        // item() — 从 0-dim 标量 tensor 取出 float 值，类似 PyTorch
        .def("item", [](const Tensor& t) {
            if (t.dim() != 0)
                throw std::runtime_error("item() only for 0-dim tensors");
            return t.unsafeGetTensorImpl()->storage_->data[t.storage_offset()];
        })
        // shallow_copy — 创建新的 Tensor 对象，共享同一个 TensorImpl（use_count+1）
        // 类似 PyTorch 中多个 Tensor handle 指向同一个 TensorImpl
        .def("shallow_copy", [](const Tensor& t) {
            return Tensor(t);  // 调用拷贝构造，TensorImplPtr 引用计数 +1
        })
        // __copy__ 支持 copy.copy(t)，同样是浅拷贝共享 TensorImpl
        .def("__copy__", [](const Tensor& t) {
            return Tensor(t);
        })
        // 算子方法 + 运算符重载（由 codegen.py 从 native_functions.yaml 自动生成）
#include "generated/tensor_bindings.inl"
        // 演示 TensorImpl 的共享（多个 Tensor 可以指向同一个 TensorImpl）
        .def("data_ptr_id", [](const Tensor& t) {
            return reinterpret_cast<uintptr_t>(t.unsafeGetTensorImpl());
        })
        // Storage 指针 ID — 用于验证 view 操作是否共享同一块内存
        .def("storage_data_ptr", [](const Tensor& t) {
            return reinterpret_cast<uintptr_t>(t.storage()->data.data());
        });

    // ============================================================
    // 模块级工厂函数
    // 对应 PyTorch: torch.ones(), torch.empty() 等
    // ============================================================
    m.def("ones", &native::ones, py::arg("shape"));
    m.def("fill", &native::fill, py::arg("shape"), py::arg("value"));
    m.def("empty", &native::empty, py::arg("shape"));

    // torch.tensor(data) — 从嵌套 list 创建 tensor
    m.def("tensor", [](py::list data) {
        std::vector<int> shape;
        std::vector<float> flat;
        infer_shape_and_flatten(data, shape, flat, 0);
        auto* impl = new TensorImpl(shape, 0.0f);
        std::copy(flat.begin(), flat.end(), impl->data_ptr());
        return Tensor(TensorImplPtr(impl));
    }, py::arg("data"));

    // 模块级算子函数（由 codegen.py 从 native_functions.yaml 自动生成）
#include "generated/module_bindings.inl"
}
