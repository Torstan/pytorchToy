"""
nn.Transformer — 完整 Transformer 实现
包含 MultiheadAttention, LayerNorm, TransformerEncoderLayer,
TransformerDecoderLayer, TransformerEncoder, TransformerDecoder, Transformer
autograd 完全在 C++ 侧完成
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
from torch.nn.linear import Linear


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps

        size = normalized_shape[-1]
        w = _C.empty([size])
        b = _C.empty([size])
        for i in range(size):
            w.flat_set(i, 1.0)
        self.weight = Parameter(Tensor(w))
        self.bias = Parameter(Tensor(b))

    def forward(self, input):
        return Tensor(_C.autograd_layer_norm(
            input._c, self.weight._c, self.bias._c, self.eps))

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        k = 1.0 / math.sqrt(embed_dim)
        self.in_proj_weight = Parameter(Tensor(_C.empty([3 * embed_dim, embed_dim])))
        _nn_C.fill_uniform(self.in_proj_weight._c, -k, k)

        if bias:
            self.in_proj_bias = Parameter(Tensor(_C.empty([3 * embed_dim])))
            _nn_C.fill_uniform(self.in_proj_bias._c, -k, k)
        else:
            self.in_proj_bias = None

        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value):
        has_bias = self.in_proj_bias is not None
        return Tensor(_C.autograd_mha(
            query._c, key._c, value._c,
            self.in_proj_weight._c,
            self.in_proj_bias._c if has_bias else _C.empty([1]),
            self.out_proj.weight._c,
            self.out_proj.bias._c if self.out_proj.bias is not None else _C.empty([1]),
            has_bias, self.num_heads))

    def __repr__(self):
        return (f"MultiheadAttention(embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads})")


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, src):
        attn_output = self.self_attn(src, src, src)
        src2 = src + attn_output
        src2 = self.norm1(src2)

        ff_output = self.linear1(src2)
        ff_output = ff_output.relu()
        ff_output = self.linear2(ff_output)
        src3 = src2 + ff_output
        src3 = self.norm2(src3)

        return src3

    def __repr__(self):
        return f"TransformerEncoderLayer()"


class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.multihead_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, tgt, memory):
        attn_output = self.self_attn(tgt, tgt, tgt)
        tgt2 = tgt + attn_output
        tgt2 = self.norm1(tgt2)

        attn_output2 = self.multihead_attn(tgt2, memory, memory)
        tgt3 = tgt2 + attn_output2
        tgt3 = self.norm2(tgt3)

        ff_output = self.linear1(tgt3)
        ff_output = ff_output.relu()
        ff_output = self.linear2(ff_output)
        tgt4 = tgt3 + ff_output
        tgt4 = self.norm3(tgt4)

        return tgt4

    def __repr__(self):
        return f"TransformerDecoderLayer()"


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        for i in range(num_layers):
            layer = TransformerEncoderLayer(
                encoder_layer.self_attn.embed_dim,
                encoder_layer.self_attn.num_heads,
                encoder_layer.linear1.out_features)
            self._modules[f'layers_{i}'] = layer
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for i in range(self.num_layers):
            output = self._modules[f'layers_{i}'](output)
        return output

    def __repr__(self):
        return f"TransformerEncoder(num_layers={self.num_layers})"


class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        for i in range(num_layers):
            layer = TransformerDecoderLayer(
                decoder_layer.self_attn.embed_dim,
                decoder_layer.self_attn.num_heads,
                decoder_layer.linear1.out_features)
            self._modules[f'layers_{i}'] = layer
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt
        for i in range(self.num_layers):
            output = self._modules[f'layers_{i}'](output, memory)
        return output

    def __repr__(self):
        return f"TransformerDecoder(num_layers={self.num_layers})"


class Transformer(Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=None, dim_feedforward=2048):
        super().__init__()
        if num_decoder_layers is None:
            num_decoder_layers = num_encoder_layers

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.d_model = d_model

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

    def __repr__(self):
        return (f"Transformer(d_model={self.d_model}, "
                f"encoder={self.encoder}, decoder={self.decoder})")
