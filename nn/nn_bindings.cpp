// ============================================================
// _nn_C Python 扩展模块
// 绑定 nn/ops.h 中的所有 C++ nn 内核到 Python
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(_nn_C, m) {
    m.doc() = "pytorchToy nn C++ kernels";

    // ============================================================
    // util 层工具函数
    // ============================================================

    // 广播操作
    m.def("broadcast_add", &util::broadcast_add);
    m.def("broadcast_sub", &util::broadcast_sub);
    m.def("broadcast_mul", &util::broadcast_mul);
    m.def("broadcast_div", &util::broadcast_div);

    // 逐元素数学函数
    m.def("elementwise_tanh", &util::elementwise_tanh);
    m.def("elementwise_exp", &util::elementwise_exp);
    m.def("elementwise_log", &util::elementwise_log);
    m.def("elementwise_sqrt", &util::elementwise_sqrt);
    m.def("elementwise_relu", &util::elementwise_relu);
    m.def("scalar_mul", &util::scalar_mul);
    m.def("scalar_add", &util::scalar_add);

    // 高级张量操作
    m.def("batched_matmul", &util::batched_matmul);
    m.def("softmax", &util::softmax, py::arg("input"), py::arg("dim"));
    m.def("log_softmax", &util::log_softmax, py::arg("input"), py::arg("dim"));
    m.def("argmax", &util::argmax, py::arg("input"), py::arg("dim"));
    m.def("sum_dim", &util::sum_dim, py::arg("input"), py::arg("dim"),
          py::arg("keepdim") = false);
    m.def("mean_dim", &util::mean_dim, py::arg("input"), py::arg("dim"),
          py::arg("keepdim") = false);
    m.def("var_dim", &util::var_dim, py::arg("input"), py::arg("dim"),
          py::arg("keepdim") = false);
    m.def("sum_all", &util::sum_all);
    m.def("cat", &util::cat, py::arg("tensors"), py::arg("dim"));
    m.def("chunk", &util::chunk, py::arg("tensor"), py::arg("n"), py::arg("dim"));
    m.def("transpose_last2", &util::transpose_last2);
    m.def("clone", &util::clone);
    m.def("zeros_like", &util::zeros_like);
    m.def("ones_like", &util::ones_like);

    // 随机数
    m.def("randint", &util::randint, py::arg("low"), py::arg("high"),
          py::arg("shape"), py::arg("seed") = 0);
    m.def("fill_randn", [](Tensor& t, unsigned seed) { util::fill_randn(t, seed); },
          py::arg("tensor"), py::arg("seed") = 0);
    m.def("fill_uniform", [](Tensor& t, float low, float high, unsigned seed) {
        util::fill_uniform(t, low, high, seed);
    }, py::arg("tensor"), py::arg("low"), py::arg("high"), py::arg("seed") = 0);

    // ============================================================
    // nn 层内核
    // ============================================================

    // Linear
    m.def("linear_forward", &nn::linear_forward,
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("has_bias"));
    m.def("linear_backward", [](const Tensor& grad_output, const Tensor& input,
                                  const Tensor& weight) {
        auto [gi, gw, gb] = nn::linear_backward(grad_output, input, weight);
        return py::make_tuple(gi, gw, gb);
    }, py::arg("grad_output"), py::arg("input"), py::arg("weight"));

    // RNN
    m.def("rnn_forward", [](const Tensor& input, const Tensor& hidden, bool has_hidden,
                              const Tensor& weight_ih, const Tensor& weight_hh,
                              const Tensor& bias_ih, const Tensor& bias_hh,
                              bool batch_first) {
        auto result = nn::rnn_forward(input, hidden, has_hidden,
                                       weight_ih, weight_hh, bias_ih, bias_hh,
                                       batch_first);
        return py::make_tuple(result.output, result.h_n);
    });

    m.def("rnn_backward", [](const Tensor& grad_output, const Tensor& grad_h_n,
                               bool has_grad_h_n,
                               const Tensor& input, const Tensor& hidden,
                               bool has_hidden,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const Tensor& bias_ih, const Tensor& bias_hh,
                               const std::vector<Tensor>& h_states,
                               bool batch_first) {
        auto result = nn::rnn_backward(grad_output, grad_h_n, has_grad_h_n,
                                        input, hidden, has_hidden,
                                        weight_ih, weight_hh, bias_ih, bias_hh,
                                        h_states, batch_first);
        return py::make_tuple(result.grad_input, result.grad_hidden,
                               result.grad_weight_ih, result.grad_weight_hh,
                               result.grad_bias_ih, result.grad_bias_hh);
    });

    // Embedding
    m.def("embedding_forward", &nn::embedding_forward);
    m.def("embedding_backward", &nn::embedding_backward);

    // LayerNorm
    m.def("layer_norm_forward", [](const Tensor& input, const Tensor& weight,
                                     const Tensor& bias, bool has_weight,
                                     bool has_bias, float eps) {
        auto result = nn::layer_norm_forward(input, weight, bias,
                                              has_weight, has_bias, eps);
        return py::make_tuple(result.output, result.mean, result.rstd);
    }, py::arg("input"), py::arg("weight"), py::arg("bias"),
       py::arg("has_weight"), py::arg("has_bias"), py::arg("eps") = 1e-5f);

    m.def("layer_norm_backward", [](const Tensor& grad_output, const Tensor& input,
                                      const Tensor& mean, const Tensor& rstd,
                                      const Tensor& weight, bool has_weight) {
        auto result = nn::layer_norm_backward(grad_output, input, mean, rstd,
                                               weight, has_weight);
        return py::make_tuple(result.grad_input, result.grad_weight, result.grad_bias);
    });

    // Attention
    m.def("scaled_dot_product_attention", [](const Tensor& Q, const Tensor& K,
                                               const Tensor& V) {
        auto result = nn::scaled_dot_product_attention(Q, K, V);
        return py::make_tuple(result.output, result.attn_weights);
    });

    m.def("scaled_dot_product_attention_backward",
          [](const Tensor& grad_output, const Tensor& Q, const Tensor& K,
             const Tensor& V, const Tensor& attn_weights) {
        auto result = nn::scaled_dot_product_attention_backward(
            grad_output, Q, K, V, attn_weights);
        return py::make_tuple(result.grad_Q, result.grad_K, result.grad_V);
    });

    // MultiheadAttention
    m.def("multihead_attention_forward",
          [](const Tensor& query, const Tensor& key, const Tensor& value,
             const Tensor& W_q, const Tensor& W_k, const Tensor& W_v, const Tensor& W_out,
             const Tensor& b_q, const Tensor& b_k, const Tensor& b_v, const Tensor& b_out,
             bool has_bias, int num_heads) {
        auto result = nn::multihead_attention_forward(
            query, key, value, W_q, W_k, W_v, W_out,
            b_q, b_k, b_v, b_out, has_bias, num_heads);
        return py::make_tuple(result.output, result.Q_proj, result.K_proj,
                               result.V_proj, result.attn_weights, result.attn_output);
    });

    // CrossEntropy
    m.def("cross_entropy_forward", [](const Tensor& input, const Tensor& target) {
        auto result = nn::cross_entropy_forward(input, target);
        return py::make_tuple(result.loss, result.softmax_output);
    });
    m.def("cross_entropy_backward", &nn::cross_entropy_backward);

    // ============================================================
    // Adam optimizer step (C++ kernel)
    // ============================================================
    m.def("adam_step", &nn::adam_step,
          py::arg("param"), py::arg("m"), py::arg("v"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"),
          py::arg("eps"), py::arg("t"));
}
