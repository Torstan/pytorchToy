"""
nn.CrossEntropyLoss — 交叉熵损失函数
分解为 log_softmax + nll_loss（与真实 PyTorch 一致）

backward 由 log_softmax 和 nll_loss 各自的 autograd 自动链式完成
"""

from torch.nn.module import Module
import torch.nn.functional as F


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        input: [N, C] logits
        target: [N] integer class indices
        """
        return F.cross_entropy(input, target)

    def __repr__(self):
        return "CrossEntropyLoss()"
