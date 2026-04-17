import torch


class Flag:
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return self.value


def fn(x, flag):
    if flag:
        return torch.sin(x)
    return torch.cos(x)


opt = torch.compile(fn, backend="eager")

x = torch.randn(3, 3)
flag = Flag(True)

out_true = opt(x, flag)
ref_true = fn(x, flag)

flag.value = False

out_false = opt(x, flag)
ref_false = fn(x, flag)

torch.testing.assert_close(out_true, ref_true)
torch.testing.assert_close(out_false, ref_false)

print("dynamo_mutable_bool_input_branch: ok")
