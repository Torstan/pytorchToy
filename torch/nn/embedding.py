"""
nn.Embedding — 嵌入查找层
output[i] = weight[indices[i]]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C
import _nn_C

from torch.tensor import Tensor
from torch.nn.module import Module
from torch.nn.parameter import Parameter
from torch.autograd_engine import record


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 初始化权重 (正态分布)
        w = _C.empty([num_embeddings, embedding_dim])
        _nn_C.fill_randn(w)
        self.weight = Parameter(Tensor(w))

    def forward(self, input):
        output_c = _nn_C.embedding_forward(input._c, self.weight._c)
        output = Tensor(output_c)

        # 记录到 tape
        saved_input = input
        saved_weight = self.weight
        saved_num = self.num_embeddings

        def backward_fn(grad_outputs):
            grad_out = grad_outputs[0]
            gw_c = _nn_C.embedding_backward(
                grad_out._c, saved_input._c, saved_num)
            return [Tensor(gw_c)]  # grad_weight

        # input 不需要梯度，只有 weight 需要
        record([output], [self.weight], backward_fn)

        return output

    def __repr__(self):
        return (f"Embedding(num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim})")
