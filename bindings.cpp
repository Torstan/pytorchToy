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
#include <pybind11/functional.h>
#include "ops.h"
#include "util/pointwise_fusion.h"
#include "autograd/variable.h"
#include "autograd/function.h"
#include "autograd/grad_ops.h"
#include "autograd/nn_grad_ops.h"
#include "autograd/engine.h"
#include "autograd/py_function.h"
#include "autograd/autograd_ops.h"

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
        .def("__setitem__", [](Tensor& t, int i, float v) {
            t.data_ptr()[i] = v;
            t.unsafeGetTensorImpl()->bump_version();
        })
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
        // 逻辑平坦索引的读/写（stride 感知）
        .def("flat_get", [](const Tensor& t, int i) {
            return t.unsafeGetTensorImpl()->read_logical(i);
        })
        .def("flat_set", [](Tensor& t, int i, float v) {
            auto* impl = t.unsafeGetTensorImpl();
            if (impl->is_contiguous()) {
                impl->data_ptr()[i] = v;
            } else {
                // 非连续时计算实际偏移
                int offset = impl->storage_offset_;
                int idx = i;
                for (int d = impl->dim() - 1; d >= 0; d--) {
                    int coord = idx % impl->sizes_[d];
                    idx /= impl->sizes_[d];
                    offset += coord * impl->strides_[d];
                }
                impl->storage_->data[offset] = v;
            }
            impl->bump_version();
        })
        // 算子方法 + 运算符重载（由 codegen.py 从 native_functions.yaml 自动生成）
#include "generated/tensor_bindings.inl"
        // 演示 TensorImpl 的共享（多个 Tensor 可以指向同一个 TensorImpl）
        .def("data_ptr_id", [](const Tensor& t) {
            return reinterpret_cast<uintptr_t>(t.unsafeGetTensorImpl());
        })
        // 版本计数 — 用于 autograd in-place 检测
        .def("_version", [](const Tensor& t) {
            return t.unsafeGetTensorImpl()->version();
        })
        .def("_bump_version", [](Tensor& t) {
            t.unsafeGetTensorImpl()->bump_version();
        })
        // Storage 指针 ID — 用于验证 view 操作是否共享同一块内存
        .def("storage_data_ptr", [](const Tensor& t) {
            return reinterpret_cast<uintptr_t>(t.storage()->data.data());
        })
        // ---- autograd 新接口 ----
        .def("has_grad", &Tensor::has_grad)
        .def("grad", [](const Tensor& t) -> py::object {
            auto* impl = t.unsafeGetTensorImpl();
            if (!impl->has_grad()) return py::none();
            // 创建一个 Tensor 包装 grad storage
            auto* gi = new TensorImpl(impl->grad_storage_, 0,
                                       std::vector<int>(impl->grad_sizes_),
                                       std::vector<int>());
            // 计算 contiguous strides
            auto& sizes = gi->sizes_;
            gi->strides_.resize(sizes.size());
            int stride = 1;
            for (int i = (int)sizes.size() - 1; i >= 0; i--) {
                gi->strides_[i] = stride;
                stride *= sizes[i];
            }
            return py::cast(Tensor(TensorImplPtr(gi)));
        })
        .def("zero_grad", &Tensor::zero_grad)
        .def("accumulate_grad", [](Tensor& t, Tensor grad) {
            Tensor grad_contig = grad;
            if (!grad_contig.is_contiguous()) {
                grad_contig = native::contiguous(grad_contig);
            }
            auto* impl = t.unsafeGetTensorImpl();
            auto* gimpl = grad_contig.unsafeGetTensorImpl();
            impl->accumulate_grad(gimpl->data_ptr(), gimpl->numel(), gimpl->sizes_);
        })
        .def("get_creator", [](const Tensor& t) -> py::object {
            auto c = t.get_creator();
            if (!c) return py::none();
            return py::cast(c);
        })
        .def("has_creator", [](const Tensor& t) {
            return t.get_creator() != nullptr;
        })
        .def("set_requires_grad", &Tensor::set_requires_grad);

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

    py::class_<pointwise::CompiledPointwiseProgram>(m, "CompiledPointwiseProgram")
        .def(
            py::init<
                std::vector<int>,
                int,
                int,
                std::vector<float>,
                std::vector<std::tuple<int, int, int, int, int, int>>,
                int,
                int>(),
            py::arg("shape"),
            py::arg("num_inputs"),
            py::arg("num_temps"),
            py::arg("consts"),
            py::arg("instructions"),
            py::arg("output_kind"),
            py::arg("output_index"))
        .def("run", &pointwise::CompiledPointwiseProgram::run);

    // ============================================================
    // Autograd 绑定
    // ============================================================

    // --- VariableImpl ---
    py::class_<VariableImpl, std::shared_ptr<VariableImpl>>(m, "VariableImpl")
        .def(py::init<Tensor, bool, bool>(),
             py::arg("data"), py::arg("requires_grad") = false, py::arg("volatile_") = false)
        .def_readwrite("data", &VariableImpl::data)
        .def_readwrite("grad", &VariableImpl::grad)
        .def_readwrite("grad_defined", &VariableImpl::grad_defined)
        .def_readwrite("requires_grad", &VariableImpl::requires_grad)
        .def_readwrite("is_volatile", &VariableImpl::is_volatile)
        .def_readwrite("output_index", &VariableImpl::output_index)
        .def("accumulate_grad", &VariableImpl::accumulate_grad)
        .def("run_hooks", &VariableImpl::run_hooks)
        .def("add_hook", [](VariableImpl& self, std::function<void(Tensor&)> hook) {
            self.hooks.push_back(std::move(hook));
        })
        // creator 的 getter/setter（shared_ptr<AutogradFunction>）
        .def("get_creator", [](VariableImpl& self) -> std::shared_ptr<AutogradFunction> {
            return self.creator;
        })
        .def("set_creator", [](VariableImpl& self, std::shared_ptr<AutogradFunction> fn) {
            self.creator = std::move(fn);
        });

    // --- AutogradFunction 基类 ---
    py::class_<AutogradFunction, std::shared_ptr<AutogradFunction>>(m, "AutogradFunction")
        .def_readwrite("num_inputs", &AutogradFunction::num_inputs)
        .def_readwrite("requires_grad", &AutogradFunction::requires_grad)
        .def("add_previous_function", [](AutogradFunction& self,
                                          std::shared_ptr<AutogradFunction> prev_fn, int idx) {
            InputInfo info;
            info.fn = std::move(prev_fn);
            info.output_index = idx;
            info.variable = nullptr;
            self.inputs.push_back(std::move(info));
        })
        .def("add_leaf_variable", [](AutogradFunction& self,
                                      std::shared_ptr<VariableImpl> var) {
            InputInfo info;
            info.fn = nullptr;
            info.output_index = 0;
            info.variable = var.get();
            self.inputs.push_back(std::move(info));
        })
        .def("apply", &AutogradFunction::apply);

    // --- PyFunction（Python 自定义 Function 的 C++ 桥接）---
    py::class_<PyFunction, AutogradFunction, std::shared_ptr<PyFunction>>(m, "PyFunction")
        .def(py::init<py::object>());

    // --- 内置 backward 函数 ---
    py::class_<BroadcastMulBackward, AutogradFunction, std::shared_ptr<BroadcastMulBackward>>(m, "MulBackward")
        .def(py::init<Tensor, Tensor, std::vector<int>, std::vector<int>>());

    py::class_<MulScalarBackward, AutogradFunction, std::shared_ptr<MulScalarBackward>>(m, "MulScalarBackward")
        .def(py::init<float>());

    py::class_<SumBackward, AutogradFunction, std::shared_ptr<SumBackward>>(m, "SumBackward")
        .def(py::init<std::vector<int>>());

    py::class_<AddBackward, AutogradFunction, std::shared_ptr<AddBackward>>(m, "AddBackward")
        .def(py::init<std::vector<int>, std::vector<int>>());

    // --- Engine ---
    m.def("engine_backward", [](std::shared_ptr<AutogradFunction> root_fn,
                                 Tensor grad_output, bool retain_graph) {
        Engine::backward(std::move(root_fn), std::move(grad_output), retain_graph);
    }, py::arg("root_fn"), py::arg("grad_output"), py::arg("retain_graph") = false);

    // ============================================================
    // C++ Autograd 算子
    // ============================================================
    m.def("autograd_add", &autograd::add);
    m.def("autograd_add_scalar", &autograd::add_scalar);
    m.def("autograd_sub", &autograd::sub);
    m.def("autograd_sub_scalar", &autograd::sub_scalar);
    m.def("autograd_mul", &autograd::mul);
    m.def("autograd_mul_scalar", &autograd::mul_scalar);
    m.def("autograd_div", &autograd::div);
    m.def("autograd_div_scalar", &autograd::div_scalar);
    m.def("autograd_neg", &autograd::neg);
    m.def("autograd_mm", &autograd::mm);
    m.def("autograd_matmul", &autograd::matmul);
    m.def("autograd_batched_matmul", &autograd::batched_matmul);
    m.def("autograd_transpose", &autograd::transpose,
          py::arg("input"), py::arg("d0"), py::arg("d1"));
    m.def("autograd_view", &autograd::view,
          py::arg("input"), py::arg("shape"));
    m.def("autograd_expand", &autograd::expand,
          py::arg("input"), py::arg("new_sizes"));
    m.def("autograd_slice", &autograd::slice,
          py::arg("input"), py::arg("dim"), py::arg("start"), py::arg("end"));
    m.def("autograd_relu", &autograd::relu);
    m.def("autograd_tanh", &autograd::tanh);
    m.def("autograd_sum", &autograd::sum);
    m.def("autograd_sum_dim", &autograd::sum_dim,
          py::arg("input"), py::arg("dim"), py::arg("keepdim") = false);
    m.def("autograd_linear", [](const Tensor& input, const Tensor& weight,
                                  py::object bias_obj,
                                  py::object packed_weight_obj) {
        bool has_bias = !bias_obj.is_none();
        Tensor bias = has_bias ? py::cast<Tensor>(bias_obj) : native::empty({1});
        if (!packed_weight_obj.is_none()) {
            Tensor packed_weight = py::cast<Tensor>(packed_weight_obj);
            return autograd::linear_packed(input, weight, packed_weight, bias, has_bias);
        }
        return autograd::linear(input, weight, bias, has_bias);
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
       py::arg("packed_weight") = py::none());
    m.def("autograd_softmax", &autograd::softmax,
          py::arg("input"), py::arg("dim"));
    m.def("autograd_log_softmax", &autograd::log_softmax,
          py::arg("input"), py::arg("dim"));
    m.def("autograd_nll_loss", &autograd::nll_loss);
    m.def("autograd_cross_entropy", &autograd::cross_entropy);
    m.def("autograd_layer_norm", [](const Tensor& input, const Tensor& weight,
                                      const Tensor& bias, float eps) {
        return autograd::layer_norm(input, weight, bias, true, true, eps);
    }, py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5f);
    m.def("autograd_embedding", &autograd::embedding);
    m.def("autograd_rnn", [](const Tensor& input, const Tensor& hidden,
                               bool has_hidden,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const Tensor& bias_ih, const Tensor& bias_hh,
                               bool batch_first) {
        auto result = autograd::rnn(
            input, hidden, has_hidden,
            weight_ih, weight_hh, bias_ih, bias_hh,
            batch_first);
        return py::make_tuple(result.output, result.h_n);
    });
    m.def("autograd_mha", [](const Tensor& query, const Tensor& key, const Tensor& value,
                               const Tensor& in_proj_weight, const Tensor& in_proj_bias,
                               const Tensor& out_proj_weight, const Tensor& out_proj_bias,
                               bool has_bias, int num_heads) {
        return autograd::multihead_attention(
            query, key, value, in_proj_weight, in_proj_bias,
            out_proj_weight, out_proj_bias, has_bias, num_heads);
    });

    // backward 入口
    m.def("autograd_backward", [](const Tensor& loss) {
        autograd::backward(loss);
    });

    // grad mode 控制
    m.def("set_grad_enabled", [](bool enabled) {
        autograd::grad_mode_enabled = enabled;
    });
    m.def("is_grad_enabled", []() {
        return autograd::grad_mode_enabled;
    });

    // --- 新增 backward 算子类绑定 (System A 兼容) ---
    py::class_<SubBackward, AutogradFunction, std::shared_ptr<SubBackward>>(m, "SubBackward")
        .def(py::init<std::vector<int>, std::vector<int>>());
    py::class_<NegBackward, AutogradFunction, std::shared_ptr<NegBackward>>(m, "NegBackward")
        .def(py::init<>());
    py::class_<ReluBackward, AutogradFunction, std::shared_ptr<ReluBackward>>(m, "ReluBackward")
        .def(py::init<Tensor>());
    py::class_<TanhBackward, AutogradFunction, std::shared_ptr<TanhBackward>>(m, "TanhBackward")
        .def(py::init<Tensor>());
    py::class_<LinearBackward, AutogradFunction, std::shared_ptr<LinearBackward>>(m, "LinearBackward")
        .def(py::init<Tensor, Tensor, bool>());
}
