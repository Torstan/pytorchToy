#pragma once

#include "function.h"
#include "../ops.h"
#include "../util/math.h"
#include "../util/tensor_ops.h"
#include "../nn/ops.h"

// ============================================================
// nn 层的 backward 算子
// ============================================================

// LinearBackward: y = linear(input, weight, bias)
class LinearBackward : public AutogradFunction {
public:
    Tensor saved_input;
    Tensor saved_weight;
    bool has_bias;

    LinearBackward(Tensor input, Tensor weight, bool hb)
        : saved_input(std::move(input)), saved_weight(std::move(weight)), has_bias(hb) {
        num_inputs = has_bias ? 3 : 2;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        auto [gi, gw, gb] = nn::linear_backward(
            grad_outputs[0], saved_input, saved_weight);
        std::vector<Tensor> grads = {gi, gw};
        if (has_bias) grads.push_back(gb);
        return grads;
    }
};

// SoftmaxBackward: y = softmax(x, dim)
class SoftmaxBackward : public AutogradFunction {
public:
    Tensor saved_output;
    int dim;

    SoftmaxBackward(Tensor output, int d)
        : saved_output(std::move(output)), dim(d) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& g = grad_outputs[0];
        // ds/dx = s * (g - sum(g * s, dim, keepdim=True))
        Tensor gs = util::broadcast_mul(g, saved_output);
        Tensor gs_sum = util::sum_dim(gs, dim, true);
        Tensor sub = util::broadcast_sub(g, gs_sum);
        return {util::broadcast_mul(saved_output, sub)};
    }
};

// LogSoftmaxBackward: y = log_softmax(x, dim)
class LogSoftmaxBackward : public AutogradFunction {
public:
    Tensor saved_output;
    int dim;

    LogSoftmaxBackward(Tensor output, int d)
        : saved_output(std::move(output)), dim(d) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& g = grad_outputs[0];
        // d(log_softmax)/dx = g - softmax * sum(g, dim, keepdim)
        Tensor softmax_out = util::elementwise_exp(saved_output);
        Tensor g_sum = util::sum_dim(g, dim, true);
        return {util::broadcast_sub(g,
                    util::broadcast_mul(softmax_out, g_sum))};
    }
};

// NllLossBackward: loss = nll_loss(log_probs, target)
class NllLossBackward : public AutogradFunction {
public:
    Tensor saved_target;
    int N, C;

    NllLossBackward(Tensor target, int n, int c)
        : saved_target(std::move(target)), N(n), C(c) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        float g_val = grad_outputs[0].unsafeGetTensorImpl()->read_logical(0);
        Tensor grad_lp = native::empty({N, C});
        float* p = grad_lp.data_ptr();
        auto* ti = saved_target.unsafeGetTensorImpl();
        for (int i = 0; i < N; i++) {
            int idx = static_cast<int>(ti->read_logical(i));
            p[i * C + idx] = -g_val / N;
        }
        return {grad_lp};
    }
};

// LayerNormBackward
class LayerNormBackward : public AutogradFunction {
public:
    Tensor saved_input;
    Tensor saved_mean;
    Tensor saved_rstd;
    Tensor saved_weight;
    bool has_weight;

    LayerNormBackward(Tensor input, Tensor mean, Tensor rstd, Tensor weight, bool hw)
        : saved_input(std::move(input)), saved_mean(std::move(mean)),
          saved_rstd(std::move(rstd)), saved_weight(std::move(weight)),
          has_weight(hw) {
        num_inputs = has_weight ? 3 : 1;  // input, weight, bias
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        auto result = nn::layer_norm_backward(
            grad_outputs[0], saved_input, saved_mean, saved_rstd,
            saved_weight, has_weight);
        std::vector<Tensor> grads = {result.grad_input};
        if (has_weight) {
            grads.push_back(result.grad_weight);
            grads.push_back(result.grad_bias);
        }
        return grads;
    }
};

// EmbeddingBackward
class EmbeddingBackward : public AutogradFunction {
public:
    Tensor saved_indices;
    int num_embeddings;

    EmbeddingBackward(Tensor indices, int ne)
        : saved_indices(std::move(indices)), num_embeddings(ne) {
        num_inputs = 1;
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        return {nn::embedding_backward(grad_outputs[0], saved_indices, num_embeddings)};
    }
};

// MhaBackward: multihead attention
class MhaBackward : public AutogradFunction {
public:
    Tensor saved_query, saved_key, saved_value;
    Tensor saved_Q_proj, saved_K_proj, saved_V_proj;
    Tensor saved_attn_weights, saved_attn_output;
    Tensor W_q, W_k, W_v, W_out;
    Tensor b_q, b_k, b_v, b_out;
    bool has_bias;
    int num_heads;

    MhaBackward(
        Tensor query, Tensor key, Tensor value,
        Tensor Q_proj, Tensor K_proj, Tensor V_proj,
        Tensor attn_weights, Tensor attn_output,
        Tensor wq, Tensor wk, Tensor wv, Tensor wout,
        Tensor bq, Tensor bk, Tensor bv, Tensor bout,
        bool hb, int nh)
        : saved_query(std::move(query)), saved_key(std::move(key)),
          saved_value(std::move(value)),
          saved_Q_proj(std::move(Q_proj)), saved_K_proj(std::move(K_proj)),
          saved_V_proj(std::move(V_proj)),
          saved_attn_weights(std::move(attn_weights)),
          saved_attn_output(std::move(attn_output)),
          W_q(std::move(wq)), W_k(std::move(wk)),
          W_v(std::move(wv)), W_out(std::move(wout)),
          b_q(std::move(bq)), b_k(std::move(bk)),
          b_v(std::move(bv)), b_out(std::move(bout)),
          has_bias(hb), num_heads(nh) {
        // inputs: in_proj_weight, [in_proj_bias], out_proj.weight, [out_proj.bias]
        num_inputs = 2 + (has_bias ? 2 : 0);
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        const Tensor& grad_out = grad_outputs[0];

        auto q_sizes = std::vector<int>(saved_query.sizes());
        int seq_q = q_sizes[0], batch = q_sizes[1], d_model = q_sizes[2];
        int d_k = d_model / num_heads;

        // 1. out_proj backward
        // 重建 merged: [seq_q, batch, d_model] from attn_output [batch, heads, seq_q, d_k]
        Tensor merged = native::empty({seq_q, batch, d_model});
        float* pm = merged.data_ptr();
        Tensor cao = saved_attn_output.is_contiguous() ?
            Tensor(saved_attn_output) : native::contiguous(saved_attn_output);
        const float* pao = cao.data_ptr();
        for (int si = 0; si < seq_q; si++)
            for (int bi = 0; bi < batch; bi++)
                for (int hi = 0; hi < num_heads; hi++)
                    for (int di = 0; di < d_k; di++)
                        pm[si * batch * d_model + bi * d_model + hi * d_k + di] =
                            pao[bi * num_heads * seq_q * d_k + hi * seq_q * d_k + si * d_k + di];

        Tensor merged_flat = native::reshape(merged, {seq_q * batch, d_model});
        Tensor go_flat = native::reshape(
            grad_out.is_contiguous() ? Tensor(grad_out) : native::contiguous(grad_out),
            {seq_q * batch, d_model});

        auto [gi_out, gw_out, gb_out] = nn::linear_backward(go_flat, merged_flat, W_out);

        // 简化处理：对 in_proj_weight 返回零梯度
        // 完整的 MHA backward 非常复杂，这里只传 out_proj 的梯度
        // 这与之前 Python 版本的行为一致
        std::vector<Tensor> grads;
        // in_proj_weight grad (zeros)
        int ipw_size = 3 * d_model;
        grads.push_back(native::empty({ipw_size, d_model}));
        if (has_bias) {
            grads.push_back(native::empty({ipw_size})); // in_proj_bias grad
        }
        grads.push_back(gw_out);  // out_proj.weight grad
        if (has_bias) {
            grads.push_back(gb_out); // out_proj.bias grad
        }
        return grads;
    }
};

// RnnBackward
class RnnBackward : public AutogradFunction {
public:
    Tensor saved_input, saved_hidden;
    bool has_hidden;
    Tensor saved_weight_ih, saved_weight_hh;
    Tensor saved_bias_ih, saved_bias_hh;
    std::vector<Tensor> saved_h_states;
    bool batch_first;

    RnnBackward(Tensor input, Tensor hidden, bool hh,
                Tensor wih, Tensor whh, Tensor bih, Tensor bhh,
                std::vector<Tensor> h_states, bool bf)
        : saved_input(std::move(input)), saved_hidden(std::move(hidden)),
          has_hidden(hh),
          saved_weight_ih(std::move(wih)), saved_weight_hh(std::move(whh)),
          saved_bias_ih(std::move(bih)), saved_bias_hh(std::move(bhh)),
          saved_h_states(std::move(h_states)), batch_first(bf) {
        num_inputs = 4; // weight_ih, weight_hh, bias_ih, bias_hh
    }

    std::vector<Tensor> apply(const std::vector<Tensor>& grad_outputs) override {
        // grad_outputs[0] = grad_output, grad_outputs[1] = grad_h_n (may not exist)
        Tensor grad_h_n = native::empty({1});
        bool has_grad_h_n = false;

        auto result = nn::rnn_backward(
            grad_outputs[0], grad_h_n, has_grad_h_n,
            saved_input, saved_hidden, has_hidden,
            saved_weight_ih, saved_weight_hh,
            saved_bias_ih, saved_bias_hh,
            saved_h_states, batch_first);

        return {result.grad_weight_ih, result.grad_weight_hh,
                result.grad_bias_ih, result.grad_bias_hh};
    }
};
