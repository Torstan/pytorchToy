"""
nn.RNN — 循环神经网络层
h_t = tanh(x_t @ W_ih^T + h_{t-1} @ W_hh^T + b_ih + b_hh)
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C
import _nn_C

from torch.tensor import Tensor
from torch.nn.module import Module
from torch.nn.parameter import Parameter
from torch.autograd_engine import record


class RNN(Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # 参数初始化 (uniform(-k, k), k = 1/sqrt(hidden_size))
        k = 1.0 / math.sqrt(hidden_size)

        w_ih = _C.empty([hidden_size, input_size])
        _nn_C.fill_uniform(w_ih, -k, k)
        self.weight_ih_l0 = Parameter(Tensor(w_ih))

        w_hh = _C.empty([hidden_size, hidden_size])
        _nn_C.fill_uniform(w_hh, -k, k)
        self.weight_hh_l0 = Parameter(Tensor(w_hh))

        b_ih = _C.empty([hidden_size])
        _nn_C.fill_uniform(b_ih, -k, k)
        self.bias_ih_l0 = Parameter(Tensor(b_ih))

        b_hh = _C.empty([hidden_size])
        _nn_C.fill_uniform(b_hh, -k, k)
        self.bias_hh_l0 = Parameter(Tensor(b_hh))

    def forward(self, input, hidden=None):
        has_hidden = hidden is not None
        hidden_c = hidden._c if has_hidden else _C.empty([1])

        output_c, h_n_c = _nn_C.rnn_forward(
            input._c, hidden_c, has_hidden,
            self.weight_ih_l0._c, self.weight_hh_l0._c,
            self.bias_ih_l0._c, self.bias_hh_l0._c,
            self.batch_first)

        output = Tensor(output_c)
        h_n = Tensor(h_n_c)

        # 保存用于 backward 的隐状态
        # 需要重新运行 forward 收集中间隐状态用于 BPTT
        saved_input = input
        saved_hidden = hidden
        saved_weight_ih = self.weight_ih_l0
        saved_weight_hh = self.weight_hh_l0
        saved_bias_ih = self.bias_ih_l0
        saved_bias_hh = self.bias_hh_l0
        saved_batch_first = self.batch_first
        saved_hidden_size = self.hidden_size

        def backward_fn(grad_outputs):
            grad_out = grad_outputs[0]

            # 重新前向收集中间隐状态
            h_states = _collect_hidden_states(
                saved_input, saved_hidden, has_hidden,
                saved_weight_ih, saved_weight_hh,
                saved_bias_ih, saved_bias_hh,
                saved_batch_first, saved_hidden_size)

            # 创建空的 grad_h_n
            sizes = list(saved_input._c.sizes())
            if saved_batch_first:
                batch = sizes[0]
            else:
                batch = sizes[1]
            grad_h_n_c = _C.empty([1, batch, saved_hidden_size])

            grads = _nn_C.rnn_backward(
                grad_out._c, grad_h_n_c, False,
                saved_input._c,
                saved_hidden._c if has_hidden else _C.empty([1]),
                has_hidden,
                saved_weight_ih._c, saved_weight_hh._c,
                saved_bias_ih._c, saved_bias_hh._c,
                h_states, saved_batch_first)

            # grads: (grad_input, grad_hidden, grad_wih, grad_whh, grad_bih, grad_bhh)
            return [Tensor(grads[0]),  # grad_input (对 input)
                    Tensor(grads[2]),  # grad_weight_ih
                    Tensor(grads[3]),  # grad_weight_hh
                    Tensor(grads[4]),  # grad_bias_ih
                    Tensor(grads[5])]  # grad_bias_hh

        inputs = [input, self.weight_ih_l0, self.weight_hh_l0,
                  self.bias_ih_l0, self.bias_hh_l0]
        record([output], inputs, backward_fn)

        return output, h_n

    def __repr__(self):
        return (f"RNN(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, "
                f"batch_first={self.batch_first})")


def _collect_hidden_states(input, hidden, has_hidden,
                            weight_ih, weight_hh, bias_ih, bias_hh,
                            batch_first, hidden_size):
    """收集 RNN forward 过程中的所有中间隐状态，用于 backward"""
    import _nn_C

    sizes = list(input._c.sizes())
    if batch_first:
        batch, seq_len, input_size = sizes
    else:
        seq_len, batch, input_size = sizes

    # h_states[0] = h_{-1}, h_states[t+1] = h_t
    h_states = []

    # 初始隐状态
    if has_hidden:
        h = _nn_C.clone(hidden._c)
        # reshape from [1, batch, hidden_size] to [batch, hidden_size]
        from torch.tensor import Tensor as T
        h_flat = T(_C.Tensor([batch, hidden_size], 0.0))
        for i in range(batch * hidden_size):
            h_flat._c.flat_set(i, h.flat_get(i))
        h_states.append(h_flat._c)
    else:
        h_states.append(_C.empty([batch, hidden_size]))

    # 前向传播收集 h_t
    ci = input._c if input._c.is_contiguous() else input._c.contiguous()
    wih_t_c = _nn_C.transpose_last2(weight_ih._c)
    whh_t_c = _nn_C.transpose_last2(weight_hh._c)
    # 确保连续
    if not wih_t_c.is_contiguous():
        wih_t_c = wih_t_c.contiguous()
    if not whh_t_c.is_contiguous():
        whh_t_c = whh_t_c.contiguous()

    h_current = h_states[0]

    for t in range(seq_len):
        h_new = _C.empty([batch, hidden_size])
        pi = ci.data_ptr_id()  # 不需要实际指针，用 C++ 内核

        # 使用简化版: 重新计算 h_t = tanh(x_t W_ih^T + h W_hh^T + b)
        # 由于已经有 rnn_forward，这里通过单步前向实现
        # 提取 x_t
        if batch_first:
            # input: [batch, seq_len, input_size]
            # 需要 slice
            x_t_list = []
            for b in range(batch):
                for k in range(input_size):
                    idx = b * seq_len * input_size + t * input_size + k
                    x_t_list.append(ci.flat_get(idx))
        else:
            x_t_list = []
            for b in range(batch):
                for k in range(input_size):
                    idx = t * batch * input_size + b * input_size + k
                    x_t_list.append(ci.flat_get(idx))

        # 计算 h_new = tanh(x_t @ W_ih^T + h @ W_hh^T + b_ih + b_hh)
        import math as _math
        for b in range(batch):
            for j in range(hidden_size):
                val = bias_ih._c.flat_get(j) + bias_hh._c.flat_get(j)
                for k in range(input_size):
                    val += x_t_list[b * input_size + k] * wih_t_c.flat_get(k * hidden_size + j)
                for k in range(hidden_size):
                    val += h_current.flat_get(b * hidden_size + k) * whh_t_c.flat_get(k * hidden_size + j)
                h_new.flat_set(b * hidden_size + j, _math.tanh(val))

        h_states.append(h_new)
        h_current = h_new

    return h_states
