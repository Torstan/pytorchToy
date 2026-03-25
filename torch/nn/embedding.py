"""
nn.Embedding — 嵌入查找层
output[i] = weight[indices[i]]
autograd 在 C++ 侧完成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import _C
import _nn_C

from torch.tensor import Tensor
from torch.nn.module import Module
from torch.nn.parameter import Parameter


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        w = _C.empty([num_embeddings, embedding_dim])
        _nn_C.fill_randn(w)
        self.weight = Parameter(Tensor(w))

    def forward(self, input):
        return Tensor(_C.autograd_embedding(input._c, self.weight._c))

    def __repr__(self):
        return (f"Embedding(num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim})")
