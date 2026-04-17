import torch


class Threshold:
    def __init__(self, value):
        self.value = value

    def __gt__(self, other):
        return self.value > other


def fn(x, threshold):
    if threshold > 0:
        return torch.sin(x)
    return torch.cos(x)


opt = torch.compile(fn, backend="eager")

x = torch.randn(3, 3)
threshold = Threshold(1)

out_pos = opt(x, threshold)
ref_pos = fn(x, threshold)

threshold.value = -1

out_neg = opt(x, threshold)
ref_neg = fn(x, threshold)

torch.testing.assert_close(out_pos, ref_pos)
torch.testing.assert_close(out_neg, ref_neg)

print("dynamo_mutable_compare_input_branch: ok")
