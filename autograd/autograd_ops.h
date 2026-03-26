#pragma once

#include "../tensor.h"
#include "../ops.h"
#include "../util/math.h"
#include "../util/tensor_ops.h"
#include "../nn/ops.h"
#include "function.h"
#include "grad_ops.h"
#include "nn_grad_ops.h"

// ============================================================
// C++ Autograd 调度层
// 每个函数：执行前向计算 → 如果需要梯度，创建 backward 节点并连接图
// ============================================================

namespace autograd {

// 全局开关（thread_local）
inline thread_local bool grad_mode_enabled = true;

inline bool needs_grad(const Tensor& a) {
    return grad_mode_enabled && (a.requires_grad() || a.get_creator());
}

inline bool needs_grad(const Tensor& a, const Tensor& b) {
    return grad_mode_enabled && (a.requires_grad() || a.get_creator() ||
                                  b.requires_grad() || b.get_creator());
}

// 连接输入到 backward 节点
inline void connect_input(AutogradFunction* fn, const Tensor& t) {
    if (t.get_creator()) {
        fn->add_input_fn(t.get_creator(), t.get_output_index());
    } else {
        fn->add_leaf(t.unsafeGetTensorImpl());
    }
}

// 设置输出的 autograd 信息
inline void set_output(Tensor& out, std::shared_ptr<AutogradFunction> fn, int idx = 0) {
    out.set_requires_grad(true);
    out.set_creator(std::move(fn), idx);
}

// ============================================================
// 基础算子
// ============================================================

inline Tensor add(const Tensor& a, const Tensor& b) {
    Tensor result = util::broadcast_add(a, b);
    if (needs_grad(a, b)) {
        auto fn = std::make_shared<AddBackward>(
            std::vector<int>(a.sizes()), std::vector<int>(b.sizes()));
        connect_input(fn.get(), a);
        connect_input(fn.get(), b);
        set_output(result, fn);
    }
    return result;
}

inline Tensor add_scalar(const Tensor& a, float s) {
    Tensor result = util::scalar_add(a, s);
    if (needs_grad(a)) {
        auto fn = std::make_shared<ScalarAddBackward>();
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor result = util::broadcast_sub(a, b);
    if (needs_grad(a, b)) {
        auto fn = std::make_shared<SubBackward>(
            std::vector<int>(a.sizes()), std::vector<int>(b.sizes()));
        connect_input(fn.get(), a);
        connect_input(fn.get(), b);
        set_output(result, fn);
    }
    return result;
}

inline Tensor sub_scalar(const Tensor& a, float s) {
    Tensor result = util::scalar_add(a, -s);
    if (needs_grad(a)) {
        auto fn = std::make_shared<SubScalarBackward>();
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor result = util::broadcast_mul(a, b);
    if (needs_grad(a, b)) {
        auto fn = std::make_shared<BroadcastMulBackward>(
            Tensor(a), Tensor(b),
            std::vector<int>(a.sizes()), std::vector<int>(b.sizes()));
        connect_input(fn.get(), a);
        connect_input(fn.get(), b);
        set_output(result, fn);
    }
    return result;
}

inline Tensor mul_scalar(const Tensor& a, float s) {
    Tensor result = util::scalar_mul(a, s);
    if (needs_grad(a)) {
        auto fn = std::make_shared<MulScalarBackward>(s);
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor div(const Tensor& a, const Tensor& b) {
    Tensor result = util::broadcast_div(a, b);
    if (needs_grad(a, b)) {
        auto fn = std::make_shared<DivBackward>(
            Tensor(a), Tensor(b),
            std::vector<int>(a.sizes()), std::vector<int>(b.sizes()));
        connect_input(fn.get(), a);
        connect_input(fn.get(), b);
        set_output(result, fn);
    }
    return result;
}

inline Tensor div_scalar(const Tensor& a, float s) {
    Tensor result = util::scalar_mul(a, 1.0f / s);
    if (needs_grad(a)) {
        auto fn = std::make_shared<DivScalarBackward>(s);
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor neg(const Tensor& a) {
    Tensor result = util::scalar_mul(a, -1.0f);
    if (needs_grad(a)) {
        auto fn = std::make_shared<NegBackward>();
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

// ============================================================
// 矩阵操作
// ============================================================

inline Tensor mm(const Tensor& a, const Tensor& b) {
    Tensor result = util::batched_matmul(a, b);
    if (needs_grad(a, b)) {
        auto fn = std::make_shared<MmBackward>(Tensor(a), Tensor(b));
        connect_input(fn.get(), a);
        connect_input(fn.get(), b);
        set_output(result, fn);
    }
    return result;
}

inline Tensor batched_matmul(const Tensor& a, const Tensor& b) {
    Tensor result = util::batched_matmul(a, b);
    if (needs_grad(a, b)) {
        auto fn = std::make_shared<BatchedMatmulBackward>(Tensor(a), Tensor(b));
        connect_input(fn.get(), a);
        connect_input(fn.get(), b);
        set_output(result, fn);
    }
    return result;
}

// matmul: 根据维度自动选择 mm 或 batched_matmul
inline Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.dim() == 2 && b.dim() == 2) {
        return mm(a, b);
    }
    return batched_matmul(a, b);
}

inline Tensor transpose(const Tensor& a, int d0, int d1) {
    Tensor result = native::transpose(a, d0, d1);
    if (needs_grad(a)) {
        auto fn = std::make_shared<TransposeBackward>(d0, d1);
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor view(const Tensor& a, std::vector<int> shape) {
    auto orig_shape = std::vector<int>(a.sizes());
    Tensor result = native::reshape(a, shape);
    if (needs_grad(a)) {
        auto fn = std::make_shared<ViewBackward>(orig_shape);
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor expand(const Tensor& a, std::vector<int> new_sizes) {
    auto orig_shape = std::vector<int>(a.sizes());
    Tensor result = native::expand(a, new_sizes);
    if (needs_grad(a)) {
        auto fn = std::make_shared<ExpandBackward>(orig_shape);
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor slice(const Tensor& a, int dim, int start, int end) {
    auto orig_shape = std::vector<int>(a.sizes());
    Tensor result = native::slice(a, dim, start, end);
    if (needs_grad(a)) {
        auto fn = std::make_shared<SliceBackward>(orig_shape, dim, start);
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

// ============================================================
// 激活函数和归约
// ============================================================

inline Tensor relu(const Tensor& a) {
    Tensor result = util::elementwise_relu(a);
    if (needs_grad(a)) {
        auto fn = std::make_shared<ReluBackward>(Tensor(a));
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

inline Tensor tanh(const Tensor& a) {
    Tensor result = util::elementwise_tanh(a);
    if (needs_grad(a)) {
        auto fn = std::make_shared<TanhBackward>(Tensor(result));
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

// sum 全局
inline Tensor sum(const Tensor& a) {
    float s = util::sum_all(a);
    Tensor result = native::empty({1});
    result.data_ptr()[0] = s;
    if (needs_grad(a)) {
        auto fn = std::make_shared<SumBackward>(std::vector<int>(a.sizes()));
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

// sum along dim
inline Tensor sum_dim(const Tensor& a, int dim, bool keepdim = false) {
    Tensor result = util::sum_dim(a, dim, keepdim);
    if (needs_grad(a)) {
        auto fn = std::make_shared<SumDimBackward>(
            std::vector<int>(a.sizes()), dim, keepdim);
        connect_input(fn.get(), a);
        set_output(result, fn);
    }
    return result;
}

// ============================================================
// nn 层算子
// ============================================================

inline Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias, bool has_bias) {
    Tensor result = nn::linear_forward(input, weight, bias, has_bias);
    bool ng = needs_grad(input, weight) || (has_bias && needs_grad(bias));
    if (ng) {
        auto fn = std::make_shared<LinearBackward>(Tensor(input), Tensor(weight), has_bias);
        connect_input(fn.get(), input);
        connect_input(fn.get(), weight);
        if (has_bias) connect_input(fn.get(), bias);
        set_output(result, fn);
    }
    return result;
}

inline Tensor softmax(const Tensor& input, int dim) {
    Tensor result = util::softmax(input, dim);
    if (needs_grad(input)) {
        auto fn = std::make_shared<SoftmaxBackward>(Tensor(result), dim);
        connect_input(fn.get(), input);
        set_output(result, fn);
    }
    return result;
}

inline Tensor log_softmax(const Tensor& input, int dim) {
    Tensor result = util::log_softmax(input, dim);
    if (needs_grad(input)) {
        auto fn = std::make_shared<LogSoftmaxBackward>(Tensor(result), dim);
        connect_input(fn.get(), input);
        set_output(result, fn);
    }
    return result;
}

inline Tensor nll_loss(const Tensor& log_probs, const Tensor& target) {
    auto sizes = std::vector<int>(log_probs.sizes());
    int N = sizes[0], C = sizes[1];

    // 计算 -mean(log_probs[i, target[i]])
    Tensor ct = target.is_contiguous() ? Tensor(target) : native::contiguous(target);
    Tensor clp = log_probs.is_contiguous() ? Tensor(log_probs) : native::contiguous(log_probs);
    const float* plp = clp.data_ptr();
    const float* pt = ct.data_ptr();

    float total = 0.0f;
    for (int i = 0; i < N; i++) {
        int idx = static_cast<int>(pt[i]);
        total += plp[i * C + idx];
    }
    float loss_val = -total / N;
    Tensor loss = native::empty({1});
    loss.data_ptr()[0] = loss_val;

    if (needs_grad(log_probs)) {
        auto fn = std::make_shared<NllLossBackward>(Tensor(target), N, C);
        connect_input(fn.get(), log_probs);
        set_output(loss, fn);
    }
    return loss;
}

inline Tensor cross_entropy(const Tensor& input, const Tensor& target) {
    return nll_loss(log_softmax(input, -1), target);
}

// LayerNorm forward with autograd
struct LayerNormResult {
    Tensor output;
    Tensor mean;
    Tensor rstd;
};

inline Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                          bool has_weight, bool has_bias, float eps) {
    auto result = nn::layer_norm_forward(input, weight, bias, has_weight, has_bias, eps);
    Tensor output = result.output;

    bool ng = needs_grad(input) ||
              (has_weight && needs_grad(weight)) ||
              (has_bias && needs_grad(bias));
    if (ng) {
        auto fn = std::make_shared<LayerNormBackward>(
            Tensor(input), result.mean, result.rstd, Tensor(weight), has_weight);
        connect_input(fn.get(), input);
        if (has_weight) {
            connect_input(fn.get(), weight);
            connect_input(fn.get(), bias);
        }
        set_output(output, fn);
    }
    return output;
}

// MHA forward with autograd
inline Tensor multihead_attention(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const Tensor& in_proj_weight, const Tensor& in_proj_bias,
        const Tensor& out_proj_weight, const Tensor& out_proj_bias,
        bool has_bias, int num_heads) {
    // 从 in_proj_weight 中切分 W_q, W_k, W_v
    auto chunks = util::chunk(in_proj_weight, 3, 0);
    Tensor W_q = chunks[0].is_contiguous() ? chunks[0] : native::contiguous(chunks[0]);
    Tensor W_k = chunks[1].is_contiguous() ? chunks[1] : native::contiguous(chunks[1]);
    Tensor W_v = chunks[2].is_contiguous() ? chunks[2] : native::contiguous(chunks[2]);

    Tensor b_q, b_k, b_v;
    if (has_bias) {
        auto bias_chunks = util::chunk(in_proj_bias, 3, 0);
        b_q = bias_chunks[0].is_contiguous() ? bias_chunks[0] : native::contiguous(bias_chunks[0]);
        b_k = bias_chunks[1].is_contiguous() ? bias_chunks[1] : native::contiguous(bias_chunks[1]);
        b_v = bias_chunks[2].is_contiguous() ? bias_chunks[2] : native::contiguous(bias_chunks[2]);
    } else {
        b_q = native::empty({1});
        b_k = native::empty({1});
        b_v = native::empty({1});
    }

    auto result = nn::multihead_attention_forward(
        query, key, value, W_q, W_k, W_v, out_proj_weight,
        b_q, b_k, b_v,
        has_bias ? out_proj_bias : native::empty({1}),
        has_bias, num_heads);

    Tensor output = result.output;

    // Connect autograd
    auto fn = std::make_shared<MhaBackward>(
        Tensor(query), Tensor(key), Tensor(value),
        result.Q_proj, result.K_proj, result.V_proj,
        result.attn_weights, result.attn_output,
        W_q, W_k, W_v, Tensor(out_proj_weight),
        b_q, b_k, b_v, Tensor(out_proj_bias),
        has_bias, num_heads);

    connect_input(fn.get(), in_proj_weight);
    if (has_bias) connect_input(fn.get(), in_proj_bias);
    connect_input(fn.get(), out_proj_weight);
    if (has_bias) connect_input(fn.get(), out_proj_bias);
    set_output(output, fn);

    return output;
}

// Embedding forward with autograd
inline Tensor embedding(const Tensor& indices, const Tensor& weight) {
    Tensor result = nn::embedding_forward(indices, weight);
    if (needs_grad(weight)) {
        int num_emb = std::vector<int>(weight.sizes())[0];
        auto fn = std::make_shared<EmbeddingBackward>(Tensor(indices), num_emb);
        connect_input(fn.get(), weight);
        set_output(result, fn);
    }
    return result;
}

// RNN forward with autograd
struct RnnResult {
    Tensor output;
    Tensor h_n;
};

inline RnnResult rnn(const Tensor& input, const Tensor& hidden, bool has_hidden,
                      const Tensor& weight_ih, const Tensor& weight_hh,
                      const Tensor& bias_ih, const Tensor& bias_hh,
                      bool batch_first) {
    // 需要保存中间 hidden states 用于 backward
    auto sizes = std::vector<int>(input.sizes());

    // 运行 forward
    auto result = nn::rnn_forward(input, hidden, has_hidden,
                                    weight_ih, weight_hh, bias_ih, bias_hh,
                                    batch_first);

    // 需要重新收集 h_states 用于 backward (forward 不暴露它们)
    // 简化: 不在 C++ autograd 中实现 RNN backward
    // RNN backward 通过 Python autograd 仍然工作
    // 这里只返回结果，不连接 autograd 图
    // TODO: 完整实现

    return {result.output, result.h_n};
}

// ============================================================
// backward 入口
// ============================================================

// 前向声明 Engine
} // namespace autograd (temporarily close)

#include "engine.h"

namespace autograd {

inline void backward(const Tensor& loss) {
    auto creator = loss.get_creator();
    if (!creator) {
        throw std::runtime_error("backward: tensor has no creator (not part of computation graph)");
    }
    Tensor grad_output = native::ones(std::vector<int>(loss.sizes()));
    Engine::backward(creator, grad_output);
}

} // namespace autograd
