#pragma once

#include "../tensor.h"
#include "../ops.h"
#include "../util/math.h"
#include "../util/tensor_ops.h"
#include <cmath>
#include <vector>
#include <tuple>
#include <stdexcept>

// ============================================================
// nn 层 C++ 内核
// 包含 Linear, RNN, Embedding, Attention, LayerNorm,
// CrossEntropy 等操作的 forward 和 backward 实现
// ============================================================

namespace nn {

// ============================================================
// Linear: y = x @ W^T + b
// input: [..., in_features]
// weight: [out_features, in_features]
// bias: [out_features] (可选)
// output: [..., out_features]
// ============================================================

inline Tensor linear_forward(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, bool has_bias) {
    // weight^T: [in_features, out_features]
    Tensor wt = util::transpose_last2(weight);
    // x @ W^T
    Tensor output = util::batched_matmul(input, wt);
    // + bias (广播)
    if (has_bias) {
        output = util::broadcast_add(output, bias);
    }
    return output;
}

// grad_output: [..., out_features]
// input: [..., in_features]
// weight: [out_features, in_features]
// 返回: (grad_input, grad_weight, grad_bias)
inline std::tuple<Tensor, Tensor, Tensor> linear_backward(
        const Tensor& grad_output, const Tensor& input, const Tensor& weight) {
    // grad_input = grad_output @ weight  [..., out] @ [out, in] → [..., in]
    Tensor grad_input = util::batched_matmul(grad_output, weight);

    // 将 input 和 grad_output 压平为 2D 计算 grad_weight
    // grad_weight = grad_output^T @ input  [out, ...] @ [..., in] → [out, in]
    auto input_sizes = std::vector<int>(input.sizes());
    auto go_sizes = std::vector<int>(grad_output.sizes());
    int in_features = input_sizes.back();
    int out_features = go_sizes.back();
    int batch_total = input.numel() / in_features;

    // 重塑为 2D
    Tensor input_2d = native::reshape(
        input.is_contiguous() ? Tensor(input) : native::contiguous(input),
        {batch_total, in_features});
    Tensor go_2d = native::reshape(
        grad_output.is_contiguous() ? Tensor(grad_output) : native::contiguous(grad_output),
        {batch_total, out_features});

    // grad_weight = go_2d^T @ input_2d  [out, batch] @ [batch, in]
    Tensor go_2d_t = native::transpose(go_2d, 0, 1);
    Tensor grad_weight = native::matmul(go_2d_t, input_2d);

    // grad_bias = sum over batch dims
    Tensor grad_bias = native::empty({out_features});
    float* pb = grad_bias.data_ptr();
    const float* pgo = go_2d.data_ptr();
    for (int j = 0; j < out_features; j++) {
        float s = 0;
        for (int i = 0; i < batch_total; i++)
            s += pgo[i * out_features + j];
        pb[j] = s;
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================
// RNN forward
// input: [batch, seq_len, input_size] (batch_first=true)
//    or  [seq_len, batch, input_size] (batch_first=false)
// hidden: [1, batch, hidden_size] or None (zeros)
// W_ih: [hidden_size, input_size]
// W_hh: [hidden_size, hidden_size]
// b_ih: [hidden_size]
// b_hh: [hidden_size]
// 返回: (output, h_n)
//   output: [batch, seq_len, hidden_size] (batch_first) or [seq_len, batch, hidden_size]
//   h_n: [1, batch, hidden_size]
// ============================================================

struct RNNResult {
    Tensor output;
    Tensor h_n;
};

inline RNNResult rnn_forward(const Tensor& input, const Tensor& hidden,
                              bool has_hidden,
                              const Tensor& weight_ih, const Tensor& weight_hh,
                              const Tensor& bias_ih, const Tensor& bias_hh,
                              bool batch_first) {
    auto sizes = std::vector<int>(input.sizes());
    int batch, seq_len, input_size;

    if (batch_first) {
        batch = sizes[0];
        seq_len = sizes[1];
        input_size = sizes[2];
    } else {
        seq_len = sizes[0];
        batch = sizes[1];
        input_size = sizes[2];
    }

    int hidden_size = std::vector<int>(weight_hh.sizes())[0];

    // 初始化隐状态
    Tensor h;
    if (has_hidden) {
        // hidden: [1, batch, hidden_size] → [batch, hidden_size]
        h = native::reshape(
            hidden.is_contiguous() ? Tensor(hidden) : native::contiguous(hidden),
            {batch, hidden_size});
    } else {
        h = native::empty({batch, hidden_size});
    }

    // 确保输入连续
    Tensor ci = input.is_contiguous() ? Tensor(input) : native::contiguous(input);

    // 预转置权重: W_ih^T [input_size, hidden_size], W_hh^T [hidden_size, hidden_size]
    Tensor wih_t = native::transpose(weight_ih, 0, 1);
    Tensor whh_t = native::transpose(weight_hh, 0, 1);

    // 确保转置后的权重连续
    wih_t = wih_t.is_contiguous() ? wih_t : native::contiguous(wih_t);
    whh_t = whh_t.is_contiguous() ? whh_t : native::contiguous(whh_t);

    // 输出收集
    Tensor output = native::empty({seq_len, batch, hidden_size});
    float* po = output.data_ptr();

    const float* pi = ci.data_ptr();
    const float* pwih = wih_t.data_ptr();
    const float* pwhh = whh_t.data_ptr();
    const float* pbih = bias_ih.data_ptr();
    const float* pbhh = bias_hh.data_ptr();

    for (int t = 0; t < seq_len; t++) {
        // h_new = tanh(x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh)
        Tensor h_new = native::empty({batch, hidden_size});
        float* ph_new = h_new.data_ptr();
        const float* ph = h.data_ptr();

        for (int b = 0; b < batch; b++) {
            // x_t[b]: 获取当前时间步当前 batch 的输入向量
            const float* px;
            int x_idx;
            if (batch_first) {
                // input[b, t, :] → offset = b * seq_len * input_size + t * input_size
                x_idx = b * seq_len * input_size + t * input_size;
            } else {
                x_idx = t * batch * input_size + b * input_size;
            }
            px = pi + x_idx;

            for (int j = 0; j < hidden_size; j++) {
                float val = pbih[j] + pbhh[j];
                // x_t @ W_ih^T
                for (int k = 0; k < input_size; k++)
                    val += px[k] * pwih[k * hidden_size + j];
                // h @ W_hh^T
                for (int k = 0; k < hidden_size; k++)
                    val += ph[b * hidden_size + k] * pwhh[k * hidden_size + j];
                ph_new[b * hidden_size + j] = std::tanh(val);
            }
        }

        h = h_new;

        // 存储输出: output[t, b, :] = h[b, :]
        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < hidden_size; j++) {
                po[t * batch * hidden_size + b * hidden_size + j] =
                    h_new.data_ptr()[b * hidden_size + j];
            }
        }
    }

    // 如果 batch_first，转换输出形状
    Tensor final_output;
    if (batch_first) {
        // [seq_len, batch, hidden_size] → [batch, seq_len, hidden_size]
        final_output = native::empty({batch, seq_len, hidden_size});
        float* pfo = final_output.data_ptr();
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seq_len; t++)
                for (int j = 0; j < hidden_size; j++)
                    pfo[b * seq_len * hidden_size + t * hidden_size + j] =
                        po[t * batch * hidden_size + b * hidden_size + j];
    } else {
        final_output = output;
    }

    // h_n: [1, batch, hidden_size]
    Tensor h_n = native::reshape(util::clone(h), {1, batch, hidden_size});

    return {final_output, h_n};
}

// RNN backward
// 返回: (grad_input, grad_hidden, grad_weight_ih, grad_weight_hh, grad_bias_ih, grad_bias_hh)
struct RNNGrads {
    Tensor grad_input;
    Tensor grad_hidden;
    Tensor grad_weight_ih;
    Tensor grad_weight_hh;
    Tensor grad_bias_ih;
    Tensor grad_bias_hh;
};

inline RNNGrads rnn_backward(const Tensor& grad_output, const Tensor& grad_h_n,
                               bool has_grad_h_n,
                               const Tensor& input, const Tensor& hidden,
                               bool has_hidden,
                               const Tensor& weight_ih, const Tensor& weight_hh,
                               const Tensor& bias_ih, const Tensor& bias_hh,
                               // saved hidden states from forward
                               const std::vector<Tensor>& h_states,
                               bool batch_first) {
    auto sizes = std::vector<int>(input.sizes());
    int batch, seq_len, input_size;
    if (batch_first) {
        batch = sizes[0]; seq_len = sizes[1]; input_size = sizes[2];
    } else {
        seq_len = sizes[0]; batch = sizes[1]; input_size = sizes[2];
    }
    int hidden_size = std::vector<int>(weight_hh.sizes())[0];

    // 确保输入连续
    Tensor ci = input.is_contiguous() ? Tensor(input) : native::contiguous(input);
    const float* pi = ci.data_ptr();

    Tensor cgo = grad_output.is_contiguous() ? Tensor(grad_output) : native::contiguous(grad_output);

    // 初始化梯度
    Tensor grad_input = native::empty(std::vector<int>(input.sizes()));
    float* pgi = grad_input.data_ptr();

    Tensor grad_wih = native::empty({hidden_size, input_size});
    Tensor grad_whh = native::empty({hidden_size, hidden_size});
    Tensor grad_bih = native::empty({hidden_size});
    Tensor grad_bhh = native::empty({hidden_size});

    // 梯度从 h_n 传来
    Tensor grad_h = native::empty({batch, hidden_size});
    float* pgh = grad_h.data_ptr();
    if (has_grad_h_n) {
        Tensor cgh = grad_h_n.is_contiguous() ? Tensor(grad_h_n) : native::contiguous(grad_h_n);
        // grad_h_n: [1, batch, hidden_size] → [batch, hidden_size]
        const float* src = cgh.data_ptr();
        for (int i = 0; i < batch * hidden_size; i++)
            pgh[i] = src[i];
    }

    const float* pwih = weight_ih.data_ptr();
    const float* pwhh = weight_hh.data_ptr();
    float* pgwih = grad_wih.data_ptr();
    float* pgwhh = grad_whh.data_ptr();
    float* pgbih = grad_bih.data_ptr();
    float* pgbhh = grad_bhh.data_ptr();

    // BPTT: 反向遍历时间步
    for (int t = seq_len - 1; t >= 0; t--) {
        // 加上 grad_output[t] 的梯度
        const float* pgo_t;
        if (batch_first) {
            // grad_output: [batch, seq_len, hidden_size]
            // 需要读 grad_output[:, t, :]
            for (int b = 0; b < batch; b++) {
                for (int j = 0; j < hidden_size; j++) {
                    pgh[b * hidden_size + j] +=
                        cgo.data_ptr()[b * seq_len * hidden_size + t * hidden_size + j];
                }
            }
        } else {
            pgo_t = cgo.data_ptr() + t * batch * hidden_size;
            for (int i = 0; i < batch * hidden_size; i++)
                pgh[i] += pgo_t[i];
        }

        // h_states[t+1] 是 h_t (forward 后的), h_states[0] 是 h_{-1}
        // h_t = tanh(...), 导数 = 1 - h_t^2
        const float* h_t = h_states[t + 1].data_ptr();
        Tensor grad_raw = native::empty({batch, hidden_size});
        float* pgr = grad_raw.data_ptr();

        for (int i = 0; i < batch * hidden_size; i++) {
            pgr[i] = pgh[i] * (1.0f - h_t[i] * h_t[i]);
        }

        // h_{t-1}
        const float* h_prev = h_states[t].data_ptr();

        // grad_weight_ih += grad_raw^T @ x_t
        // grad_weight_hh += grad_raw^T @ h_{t-1}
        // grad_bias += sum(grad_raw, batch_dim)
        for (int b = 0; b < batch; b++) {
            // x_t[b]
            const float* px;
            if (batch_first) {
                px = pi + b * seq_len * input_size + t * input_size;
            } else {
                px = pi + t * batch * input_size + b * input_size;
            }

            for (int j = 0; j < hidden_size; j++) {
                float g = pgr[b * hidden_size + j];
                // grad_wih[j, k] += g * x[k]
                for (int k = 0; k < input_size; k++)
                    pgwih[j * input_size + k] += g * px[k];
                // grad_whh[j, k] += g * h_prev[k]
                for (int k = 0; k < hidden_size; k++)
                    pgwhh[j * hidden_size + k] += g * h_prev[b * hidden_size + k];
                pgbih[j] += g;
                pgbhh[j] += g;
            }
        }

        // grad_input[t] = grad_raw @ W_ih  [batch, hidden] @ [hidden, input] → [batch, input]
        for (int b = 0; b < batch; b++) {
            float* pgi_t;
            if (batch_first) {
                pgi_t = pgi + b * seq_len * input_size + t * input_size;
            } else {
                pgi_t = pgi + t * batch * input_size + b * input_size;
            }
            for (int k = 0; k < input_size; k++) {
                float s = 0;
                for (int j = 0; j < hidden_size; j++)
                    s += pgr[b * hidden_size + j] * pwih[j * input_size + k];
                pgi_t[k] = s;
            }
        }

        // 传递到 h_{t-1}: grad_h = grad_raw @ W_hh  [batch, hidden] @ [hidden, hidden]
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < hidden_size; k++) {
                float s = 0;
                for (int j = 0; j < hidden_size; j++)
                    s += pgr[b * hidden_size + j] * pwhh[j * hidden_size + k];
                pgh[b * hidden_size + k] = s;
            }
        }
    }

    // grad_hidden: [1, batch, hidden_size]
    Tensor grad_hidden = native::reshape(grad_h, {1, batch, hidden_size});

    return {grad_input, grad_hidden, grad_wih, grad_whh, grad_bih, grad_bhh};
}

// ============================================================
// Embedding: output[i] = weight[indices[i]]
// indices: 任意形状，值为整数索引 (float 存储)
// weight: [num_embeddings, embedding_dim]
// output: [*indices_shape, embedding_dim]
// ============================================================

inline Tensor embedding_forward(const Tensor& indices, const Tensor& weight) {
    auto idx_sizes = std::vector<int>(indices.sizes());
    auto w_sizes = std::vector<int>(weight.sizes());
    int emb_dim = w_sizes[1];

    std::vector<int> result_shape = idx_sizes;
    result_shape.push_back(emb_dim);

    int n_indices = indices.numel();
    Tensor result = native::empty(result_shape);
    float* pr = result.data_ptr();

    Tensor ci = indices.is_contiguous() ? Tensor(indices) : native::contiguous(indices);
    Tensor cw = weight.is_contiguous() ? Tensor(weight) : native::contiguous(weight);
    const float* pi = ci.data_ptr();
    const float* pw = cw.data_ptr();

    for (int i = 0; i < n_indices; i++) {
        int idx = static_cast<int>(pi[i]);
        for (int j = 0; j < emb_dim; j++)
            pr[i * emb_dim + j] = pw[idx * emb_dim + j];
    }
    return result;
}

inline Tensor embedding_backward(const Tensor& grad_output, const Tensor& indices,
                                   int num_embeddings) {
    auto w_shape = std::vector<int>{num_embeddings,
        (int)(grad_output.numel() / indices.numel())};
    int emb_dim = w_shape[1];
    Tensor grad_weight = native::empty(w_shape);
    float* pgw = grad_weight.data_ptr();

    Tensor ci = indices.is_contiguous() ? Tensor(indices) : native::contiguous(indices);
    Tensor cg = grad_output.is_contiguous() ? Tensor(grad_output) : native::contiguous(grad_output);
    const float* pi = ci.data_ptr();
    const float* pg = cg.data_ptr();
    int n_indices = indices.numel();

    for (int i = 0; i < n_indices; i++) {
        int idx = static_cast<int>(pi[i]);
        for (int j = 0; j < emb_dim; j++)
            pgw[idx * emb_dim + j] += pg[i * emb_dim + j];
    }
    return grad_weight;
}

// ============================================================
// LayerNorm
// input: [..., normalized_shape]
// weight, bias: [normalized_shape]
// ============================================================

struct LayerNormResult {
    Tensor output;
    Tensor mean;
    Tensor rstd;   // 1/sqrt(var + eps)
};

inline LayerNormResult layer_norm_forward(const Tensor& input,
                                           const Tensor& weight, const Tensor& bias,
                                           bool has_weight, bool has_bias,
                                           float eps) {
    auto sizes = std::vector<int>(input.sizes());
    int ndim = sizes.size();
    int norm_size = sizes[ndim - 1];
    int outer = input.numel() / norm_size;

    Tensor ct = input.is_contiguous() ? Tensor(input) : native::contiguous(input);
    Tensor result = native::empty(sizes);
    Tensor mean_t = native::empty({outer});
    Tensor rstd_t = native::empty({outer});

    const float* src = ct.data_ptr();
    float* dst = result.data_ptr();
    float* pm = mean_t.data_ptr();
    float* pr = rstd_t.data_ptr();

    const float* pw = has_weight ? weight.data_ptr() : nullptr;
    const float* pb = has_bias ? bias.data_ptr() : nullptr;

    for (int i = 0; i < outer; i++) {
        const float* row = src + i * norm_size;
        float* out_row = dst + i * norm_size;

        // 计算均值
        float mean = 0;
        for (int j = 0; j < norm_size; j++) mean += row[j];
        mean /= norm_size;

        // 计算方差
        float var = 0;
        for (int j = 0; j < norm_size; j++) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= norm_size;

        float rstd = 1.0f / std::sqrt(var + eps);

        pm[i] = mean;
        pr[i] = rstd;

        // 归一化 + 仿射
        for (int j = 0; j < norm_size; j++) {
            float normalized = (row[j] - mean) * rstd;
            if (pw) normalized *= pw[j];
            if (pb) normalized += pb[j];
            out_row[j] = normalized;
        }
    }

    return {result, mean_t, rstd_t};
}

struct LayerNormGrads {
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
};

inline LayerNormGrads layer_norm_backward(const Tensor& grad_output,
                                            const Tensor& input,
                                            const Tensor& mean, const Tensor& rstd,
                                            const Tensor& weight, bool has_weight) {
    auto sizes = std::vector<int>(input.sizes());
    int norm_size = sizes.back();
    int outer = input.numel() / norm_size;

    Tensor ci = input.is_contiguous() ? Tensor(input) : native::contiguous(input);
    Tensor cgo = grad_output.is_contiguous() ? Tensor(grad_output) : native::contiguous(grad_output);

    Tensor grad_input = native::empty(sizes);
    Tensor grad_weight = native::empty({norm_size});
    Tensor grad_bias = native::empty({norm_size});

    const float* pi = ci.data_ptr();
    const float* pgo = cgo.data_ptr();
    const float* pm = mean.data_ptr();
    const float* pr = rstd.data_ptr();
    const float* pw = has_weight ? weight.data_ptr() : nullptr;
    float* pgi = grad_input.data_ptr();
    float* pgw = grad_weight.data_ptr();
    float* pgb = grad_bias.data_ptr();

    for (int i = 0; i < outer; i++) {
        const float* row = pi + i * norm_size;
        const float* grow = pgo + i * norm_size;
        float* girow = pgi + i * norm_size;
        float m = pm[i];
        float r = pr[i];

        // 计算 ds 和 db (layernorm backward 的中间量)
        float ds = 0, db_val = 0;
        for (int j = 0; j < norm_size; j++) {
            float x_hat = (row[j] - m) * r;
            float g = grow[j] * (pw ? pw[j] : 1.0f);
            ds += g * x_hat;
            db_val += g;
        }

        for (int j = 0; j < norm_size; j++) {
            float x_hat = (row[j] - m) * r;
            float g = grow[j] * (pw ? pw[j] : 1.0f);
            girow[j] = r * (g - (x_hat * ds + db_val) / norm_size);
        }

        // grad_weight, grad_bias
        for (int j = 0; j < norm_size; j++) {
            float x_hat = (row[j] - m) * r;
            pgw[j] += grow[j] * x_hat;
            pgb[j] += grow[j];
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================
// Scaled Dot-Product Attention
// Q: [batch, num_heads, seq_q, d_k]
// K: [batch, num_heads, seq_k, d_k]
// V: [batch, num_heads, seq_k, d_v]
// output: [batch, num_heads, seq_q, d_v]
// ============================================================

struct AttentionResult {
    Tensor output;
    Tensor attn_weights; // softmax weights for backward
};

inline AttentionResult scaled_dot_product_attention(
        const Tensor& Q, const Tensor& K, const Tensor& V) {
    auto q_sizes = std::vector<int>(Q.sizes());
    int d_k = q_sizes.back();
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // scores = Q @ K^T / sqrt(d_k)
    Tensor Kt = util::transpose_last2(K);
    Kt = Kt.is_contiguous() ? Kt : native::contiguous(Kt);
    Tensor scores = util::batched_matmul(Q, Kt);
    scores = util::scalar_mul(scores, scale);

    // softmax over last dim
    int last_dim = scores.dim() - 1;
    Tensor attn_weights = util::softmax(scores, last_dim);

    // output = attn_weights @ V
    Tensor output = util::batched_matmul(attn_weights, V);

    return {output, attn_weights};
}

// Attention backward
struct AttentionGrads {
    Tensor grad_Q;
    Tensor grad_K;
    Tensor grad_V;
};

inline AttentionGrads scaled_dot_product_attention_backward(
        const Tensor& grad_output, const Tensor& Q, const Tensor& K,
        const Tensor& V, const Tensor& attn_weights) {
    auto q_sizes = std::vector<int>(Q.sizes());
    int d_k = q_sizes.back();
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // grad_V = attn_weights^T @ grad_output
    Tensor attn_t = util::transpose_last2(attn_weights);
    attn_t = attn_t.is_contiguous() ? attn_t : native::contiguous(attn_t);
    Tensor grad_V = util::batched_matmul(attn_t, grad_output);

    // grad_attn = grad_output @ V^T
    Tensor Vt = util::transpose_last2(V);
    Vt = Vt.is_contiguous() ? Vt : native::contiguous(Vt);
    Tensor grad_attn = util::batched_matmul(grad_output, Vt);

    // softmax backward: grad_scores = attn_weights * (grad_attn - sum(grad_attn * attn_weights, dim=-1, keepdim))
    auto aw_sizes = std::vector<int>(attn_weights.sizes());
    int ndim = aw_sizes.size();
    int last_size = aw_sizes[ndim - 1];
    int outer_total = attn_weights.numel() / last_size;

    Tensor grad_scores = native::empty(aw_sizes);
    Tensor caw = attn_weights.is_contiguous() ? Tensor(attn_weights) : native::contiguous(attn_weights);
    Tensor cga = grad_attn.is_contiguous() ? Tensor(grad_attn) : native::contiguous(grad_attn);
    const float* paw = caw.data_ptr();
    const float* pga = cga.data_ptr();
    float* pgs = grad_scores.data_ptr();

    for (int i = 0; i < outer_total; i++) {
        float dot = 0;
        for (int j = 0; j < last_size; j++)
            dot += pga[i * last_size + j] * paw[i * last_size + j];
        for (int j = 0; j < last_size; j++)
            pgs[i * last_size + j] = paw[i * last_size + j] * (pga[i * last_size + j] - dot);
    }

    // scale
    grad_scores = util::scalar_mul(grad_scores, scale);

    // grad_Q = grad_scores @ K
    Tensor grad_Q = util::batched_matmul(grad_scores, K);

    // grad_K = grad_scores^T @ Q
    Tensor grad_scores_t = util::transpose_last2(grad_scores);
    grad_scores_t = grad_scores_t.is_contiguous() ? grad_scores_t : native::contiguous(grad_scores_t);
    Tensor grad_K = util::batched_matmul(grad_scores_t, Q);

    return {grad_Q, grad_K, grad_V};
}

// ============================================================
// MultiheadAttention forward
// query: [seq_q, batch, d_model]
// key: [seq_k, batch, d_model]
// value: [seq_k, batch, d_model]
// W_q, W_k, W_v: [d_model, d_model]
// W_out: [d_model, d_model]
// b_q, b_k, b_v: [d_model]
// b_out: [d_model]
// num_heads: int
// 输出: [seq_q, batch, d_model]
// ============================================================

struct MHAResult {
    Tensor output;
    // 保存用于 backward 的中间结果
    Tensor Q_proj;    // [batch, num_heads, seq_q, d_k]
    Tensor K_proj;    // [batch, num_heads, seq_k, d_k]
    Tensor V_proj;    // [batch, num_heads, seq_k, d_k]
    Tensor attn_weights;
    Tensor attn_output; // [batch, num_heads, seq_q, d_k] before reshape
};

inline MHAResult multihead_attention_forward(
        const Tensor& query, const Tensor& key, const Tensor& value,
        const Tensor& W_q, const Tensor& W_k, const Tensor& W_v, const Tensor& W_out,
        const Tensor& b_q, const Tensor& b_k, const Tensor& b_v, const Tensor& b_out,
        bool has_bias,
        int num_heads) {
    auto q_sizes = std::vector<int>(query.sizes());
    int seq_q = q_sizes[0], batch = q_sizes[1], d_model = q_sizes[2];
    auto k_sizes = std::vector<int>(key.sizes());
    int seq_k = k_sizes[0];
    int d_k = d_model / num_heads;

    // 线性投影: [seq, batch, d_model] → reshape → [seq*batch, d_model] → matmul → reshape
    Tensor q_flat = native::reshape(
        query.is_contiguous() ? Tensor(query) : native::contiguous(query),
        {seq_q * batch, d_model});
    Tensor k_flat = native::reshape(
        key.is_contiguous() ? Tensor(key) : native::contiguous(key),
        {seq_k * batch, d_model});
    Tensor v_flat = native::reshape(
        value.is_contiguous() ? Tensor(value) : native::contiguous(value),
        {seq_k * batch, d_model});

    Tensor Wq_t = native::transpose(W_q, 0, 1);
    Tensor Wk_t = native::transpose(W_k, 0, 1);
    Tensor Wv_t = native::transpose(W_v, 0, 1);
    Wq_t = Wq_t.is_contiguous() ? Wq_t : native::contiguous(Wq_t);
    Wk_t = Wk_t.is_contiguous() ? Wk_t : native::contiguous(Wk_t);
    Wv_t = Wv_t.is_contiguous() ? Wv_t : native::contiguous(Wv_t);

    Tensor Q_proj = native::matmul(q_flat, Wq_t);
    Tensor K_proj = native::matmul(k_flat, Wk_t);
    Tensor V_proj = native::matmul(v_flat, Wv_t);

    if (has_bias) {
        Q_proj = util::broadcast_add(Q_proj, b_q);
        K_proj = util::broadcast_add(K_proj, b_k);
        V_proj = util::broadcast_add(V_proj, b_v);
    }

    // 重塑为多头: [seq*batch, d_model] → [seq, batch, num_heads, d_k]
    //           → [batch, num_heads, seq, d_k]
    Q_proj = native::reshape(Q_proj, {seq_q, batch, num_heads, d_k});
    K_proj = native::reshape(K_proj, {seq_k, batch, num_heads, d_k});
    V_proj = native::reshape(V_proj, {seq_k, batch, num_heads, d_k});

    // 转置: [seq, batch, heads, d_k] → [batch, heads, seq, d_k]
    // 通过手动重排实现
    auto transpose_to_bhs = [](const Tensor& t, int s, int b, int h, int d) -> Tensor {
        Tensor ct = t.is_contiguous() ? Tensor(t) : native::contiguous(t);
        const float* src = ct.data_ptr();
        Tensor result = native::empty({b, h, s, d});
        float* dst = result.data_ptr();
        for (int bi = 0; bi < b; bi++)
            for (int hi = 0; hi < h; hi++)
                for (int si = 0; si < s; si++)
                    for (int di = 0; di < d; di++)
                        dst[bi * h * s * d + hi * s * d + si * d + di] =
                            src[si * b * h * d + bi * h * d + hi * d + di];
        return result;
    };

    Tensor Q4d = transpose_to_bhs(Q_proj, seq_q, batch, num_heads, d_k);
    Tensor K4d = transpose_to_bhs(K_proj, seq_k, batch, num_heads, d_k);
    Tensor V4d = transpose_to_bhs(V_proj, seq_k, batch, num_heads, d_k);

    // Scaled dot-product attention
    auto attn_result = scaled_dot_product_attention(Q4d, K4d, V4d);
    Tensor attn_out = attn_result.output;  // [batch, heads, seq_q, d_k]

    // 合并多头: [batch, heads, seq_q, d_k] → [seq_q, batch, d_model]
    Tensor merged = native::empty({seq_q, batch, d_model});
    float* pm = merged.data_ptr();
    Tensor cao = attn_out.is_contiguous() ? Tensor(attn_out) : native::contiguous(attn_out);
    const float* pao = cao.data_ptr();
    for (int si = 0; si < seq_q; si++)
        for (int bi = 0; bi < batch; bi++)
            for (int hi = 0; hi < num_heads; hi++)
                for (int di = 0; di < d_k; di++)
                    pm[si * batch * d_model + bi * d_model + hi * d_k + di] =
                        pao[bi * num_heads * seq_q * d_k + hi * seq_q * d_k + si * d_k + di];

    // 输出投影: [seq_q * batch, d_model] @ W_out^T
    Tensor merged_flat = native::reshape(merged, {seq_q * batch, d_model});
    Tensor Wout_t = native::transpose(W_out, 0, 1);
    Wout_t = Wout_t.is_contiguous() ? Wout_t : native::contiguous(Wout_t);
    Tensor output = native::matmul(merged_flat, Wout_t);
    if (has_bias) {
        output = util::broadcast_add(output, b_out);
    }
    output = native::reshape(output, {seq_q, batch, d_model});

    return {output, Q4d, K4d, V4d, attn_result.attn_weights, attn_out};
}

// ============================================================
// CrossEntropy: loss = -log(softmax(input))[target]
// input: [N, C] (logits)
// target: [N] (integer class indices, stored as float)
// ============================================================

struct CrossEntropyResult {
    Tensor loss;           // scalar [1]
    Tensor softmax_output; // [N, C] for backward
};

inline CrossEntropyResult cross_entropy_forward(const Tensor& input, const Tensor& target) {
    auto sizes = std::vector<int>(input.sizes());
    int N = sizes[0], C = sizes[1];

    Tensor ci = input.is_contiguous() ? Tensor(input) : native::contiguous(input);
    Tensor ct = target.is_contiguous() ? Tensor(target) : native::contiguous(target);
    const float* pt = ct.data_ptr();

    // softmax (数值稳定)
    Tensor sm = util::softmax(ci, 1);
    const float* psm = sm.data_ptr();

    // NLL loss
    float total_loss = 0;
    for (int i = 0; i < N; i++) {
        int cls = static_cast<int>(pt[i]);
        float p = psm[i * C + cls];
        total_loss -= std::log(p + 1e-12f);
    }
    total_loss /= N;

    Tensor loss = native::empty({1});
    loss.data_ptr()[0] = total_loss;

    return {loss, sm};
}

inline Tensor cross_entropy_backward(const Tensor& grad_output,
                                      const Tensor& softmax_output,
                                      const Tensor& target) {
    auto sizes = std::vector<int>(softmax_output.sizes());
    int N = sizes[0], C = sizes[1];

    float grad_scale = grad_output.data_ptr()[0] / N;

    Tensor grad_input = util::clone(softmax_output);
    float* pg = grad_input.data_ptr();
    Tensor ct = target.is_contiguous() ? Tensor(target) : native::contiguous(target);
    const float* pt = ct.data_ptr();

    for (int i = 0; i < N; i++) {
        int cls = static_cast<int>(pt[i]);
        pg[i * C + cls] -= 1.0f;
        for (int j = 0; j < C; j++)
            pg[i * C + j] *= grad_scale;
    }
    return grad_input;
}

} // namespace nn
