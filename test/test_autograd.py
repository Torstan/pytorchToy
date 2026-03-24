#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable, Function

tests_passed = 0
tests_failed = 0


def check(name, condition, msg=""):
    global tests_passed, tests_failed
    status = "PASS" if condition else "FAIL"
    print("  %-55s [%s]%s" % (name, status, ("  " + msg) if not condition else ""))
    if condition:
        tests_passed += 1
    else:
        tests_failed += 1


def to_scalar(v):
    """Extract a Python float from a Tensor / Variable / number."""
    if hasattr(v, 'data'):
        v = v.data
    while hasattr(v, 'dim') and v.dim() > 0:
        v = v[0]
    if hasattr(v, 'item'):
        return v.item()
    return float(v)


def float_eq(a, b, eps=1e-5):
    return abs(to_scalar(a) - to_scalar(b)) < eps


# ============================================================
#  Custom Functions
# ============================================================

class Square(Function):
    """y = x^2, dy/dx = 2x"""
    def forward(self, x):
        self.save_for_backward(x)
        return x * x

    def backward(self, grad_output):
        x, = self.saved_tensors
        return 2 * x * grad_output


class MulConstant(Function):
    """y = x * scalar, dy/dx = scalar"""
    def forward(self, x):
        self.scale = self.scale  # already set before calling
        return x * self.scale

    def backward(self, grad_output):
        return grad_output * self.scale


class Add(Function):
    """y = x1 + x2"""
    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, grad_output):
        return grad_output, grad_output


class Linear(Function):
    """y = x @ w + b"""
    def forward(self, x, w, b):
        self.save_for_backward(x, w)
        output = x.mm(w)
        output += b.unsqueeze(0).expand_as(output)
        return output

    def backward(self, grad_output):
        x, w = self.saved_tensors
        grad_x = grad_output.mm(w.t())
        grad_w = x.t().mm(grad_output)
        grad_b = grad_output.sum(0)
        return grad_x, grad_w, grad_b


class ReLU(Function):
    """y = max(0, x)"""
    def forward(self, x):
        self.save_for_backward(x)
        return x.clamp(min=0)

    def backward(self, grad_output):
        x, = self.saved_tensors
        mask = x.gt(0).float()
        return grad_output * mask


# ============================================================
#  Test cases
# ============================================================

# --- 1. Variable basics ---
print("[Variable basics]")

x = Variable(torch.FloatTensor([3.0]), requires_grad=True)
check("Variable creation", x.data[0] == 3.0)
check("requires_grad=True", x.requires_grad)

y = Variable(torch.FloatTensor([1.0]), requires_grad=False)
check("requires_grad=False", not y.requires_grad)

v = Variable(torch.FloatTensor([1.0]), volatile=True)
check("volatile=True", v.volatile)

# --- 2. Simple backward: y = x^2 ---
print("\n[Square: y = x^2]")

x = Variable(torch.FloatTensor([3.0]), requires_grad=True)
y = Square()(x)
check("forward: y = 9.0", float_eq(y.data[0], 9.0))

y.backward()
check("backward: dy/dx = 2*3 = 6.0", float_eq(x.grad.data[0], 6.0))

# x = -2
x2 = Variable(torch.FloatTensor([-2.0]), requires_grad=True)
y2 = Square()(x2)
y2.backward()
check("backward: dy/dx = 2*(-2) = -4.0", float_eq(x2.grad.data[0], -4.0))

# --- 3. Chain rule: y = (x^2)^2 = x^4, dy/dx = 4x^3 ---
print("\n[Chain rule: y = (x^2)^2 = x^4]")

x = Variable(torch.FloatTensor([2.0]), requires_grad=True)
h = Square()(x)   # h = x^2 = 4
y = Square()(h)   # y = h^2 = 16
check("forward: y = 16.0", float_eq(y.data[0], 16.0))

y.backward()
# dy/dx = dy/dh * dh/dx = 2h * 2x = 2*4 * 2*2 = 32
check("backward: dy/dx = 4*x^3 = 32.0", float_eq(x.grad.data[0], 32.0))

# --- 4. Add function: y = x1 + x2 ---
print("\n[Add: y = x1 + x2]")

x1 = Variable(torch.FloatTensor([3.0]), requires_grad=True)
x2 = Variable(torch.FloatTensor([5.0]), requires_grad=True)
y = Add()(x1, x2)
check("forward: y = 8.0", float_eq(y.data[0], 8.0))

y.backward()
check("backward: dy/dx1 = 1.0", float_eq(x1.grad.data[0], 1.0))
check("backward: dy/dx2 = 1.0", float_eq(x2.grad.data[0], 1.0))

# --- 5. Composite: y = (x1 + x2)^2, dy/dx1 = dy/dx2 = 2(x1+x2) ---
print("\n[Composite: y = (x1 + x2)^2]")

x1 = Variable(torch.FloatTensor([2.0]), requires_grad=True)
x2 = Variable(torch.FloatTensor([3.0]), requires_grad=True)
s = Add()(x1, x2)       # s = 5
y = Square()(s)          # y = 25
check("forward: y = 25.0", float_eq(y.data[0], 25.0))

y.backward()
check("backward: dy/dx1 = 2*(2+3) = 10.0", float_eq(x1.grad.data[0], 10.0))
check("backward: dy/dx2 = 2*(2+3) = 10.0", float_eq(x2.grad.data[0], 10.0))

# --- 6. Multi-dimensional tensor ---
print("\n[Multi-dimensional: y = x^2, x is 2x3]")

x = Variable(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]), requires_grad=True)
y = Square()(x)
check("forward shape", list(y.data.size()) == [2, 3])
check("forward values", float_eq(y.data[0][0], 1.0) and float_eq(y.data[1][2], 36.0))

loss = y.sum()
loss.backward()
check("backward: grad = 2*x", float_eq(x.grad.data[0][1], 4.0) and float_eq(x.grad.data[1][0], 8.0))

# --- 7. Linear function: y = x @ w + b ---
print("\n[Linear: y = x @ w + b]")

x = Variable(torch.FloatTensor([[1, 2], [3, 4]]), requires_grad=True)
w = Variable(torch.FloatTensor([[0.5, -0.5], [0.5, -0.5]]), requires_grad=True)
b = Variable(torch.FloatTensor([1.0, -1.0]), requires_grad=True)
y = Linear()(x, w, b)
# y = [[1*0.5+2*0.5+1, 1*-0.5+2*-0.5-1], [3*0.5+4*0.5+1, 3*-0.5+4*-0.5-1]]
#   = [[2.5, -2.5], [4.5, -4.5]]
check("forward [0,0] = 2.5", float_eq(y.data[0][0], 2.5))
check("forward [1,1] = -4.5", float_eq(y.data[1][1], -4.5))

loss = y.sum()
loss.backward()
gb = b.grad.data.view(-1)  # flatten in case sum(0) keeps dim
check("grad_b = [2, 2]", float_eq(gb[0], 2.0) and float_eq(gb[1], 2.0))
check("grad_w shape", list(w.grad.data.size()) == [2, 2])
check("grad_x shape", list(x.grad.data.size()) == [2, 2])

# --- 8. ReLU ---
print("\n[ReLU: y = max(0, x)]")

x = Variable(torch.FloatTensor([-2, -1, 0, 1, 2]), requires_grad=True)
y = ReLU()(x)
check("forward", float_eq(y.data[0], 0) and float_eq(y.data[3], 1) and float_eq(y.data[4], 2))

loss = y.sum()
loss.backward()
check("backward: grad = [0,0,0,1,1]",
      float_eq(x.grad.data[0], 0) and float_eq(x.grad.data[3], 1) and float_eq(x.grad.data[4], 1))

# --- 9. Volatile mode ---
print("\n[Volatile mode]")

x = Variable(torch.FloatTensor([3.0]), volatile=True)
y = Square()(x)
check("volatile propagates", y.volatile)
check("volatile: creator is None", y.creator is None)

# --- 10. retain_variables ---
print("\n[retain_variables]")

x = Variable(torch.FloatTensor([3.0]), requires_grad=True)
y = Square()(x)
y.backward(retain_variables=True)
grad1 = x.grad.data[0]
y.backward()
grad2 = x.grad.data[0]
check("first backward: grad = 6.0", float_eq(grad1, 6.0))
check("second backward: grad accumulated = 12.0", float_eq(grad2, 12.0))

# --- 11. Grad with non-unit grad_output ---
print("\n[Non-unit grad_output]")

x = Variable(torch.FloatTensor([3.0]), requires_grad=True)
y = Square()(x)
y.backward(torch.FloatTensor([2.0]))
check("grad_output=2: grad = 2*2*3 = 12.0", float_eq(x.grad.data[0], 12.0))

# --- 12. no grad path ---
print("\n[No grad path]")

x = Variable(torch.FloatTensor([3.0]), requires_grad=False)
y = Square()(x)
check("no grad: requires_grad=False", not y.requires_grad)

# --- 13. MLP: Linear -> ReLU -> Linear ---
print("\n[MLP: Linear -> ReLU -> Linear]")

torch.manual_seed(42)
x = Variable(torch.randn(2, 3), requires_grad=True)
w1 = Variable(torch.randn(3, 4), requires_grad=True)
b1 = Variable(torch.zeros(4), requires_grad=True)
w2 = Variable(torch.randn(4, 1), requires_grad=True)
b2 = Variable(torch.zeros(1), requires_grad=True)

h = Linear()(x, w1, b1)
h = ReLU()(h)
out = Linear()(h, w2, b2)
loss = (out * out).sum()
loss.backward()

check("x.grad exists", x.grad is not None)
check("w1.grad exists", w1.grad is not None)
check("w2.grad exists", w2.grad is not None)
check("b1.grad exists", b1.grad is not None)
check("b2.grad exists", b2.grad is not None)
check("x.grad no NaN", not (x.grad.data != x.grad.data).any())
check("w1.grad no NaN", not (w1.grad.data != w1.grad.data).any())

# --- 14. Hook test ---
print("\n[Hooks]")

hook_called = [False]
hook_grad = [None]

def my_hook(grad):
    hook_called[0] = True
    hook_grad[0] = grad.data.clone()

x = Variable(torch.FloatTensor([4.0]), requires_grad=True)
x.register_hook(my_hook)
y = Square()(x)
y.backward()
check("hook called", hook_called[0])
check("hook grad = 8.0", float_eq(hook_grad[0][0], 8.0))

# --- 15. In-place detection via save_for_backward ---
print("\n[In-place modification detection]")

# Case 1: in-place 修改 saved tensor 应报错
x = Variable(torch.FloatTensor([3.0]), requires_grad=True)
y = Square()(x)  # Square saves x for backward
x.data[0] = 999.0  # in-place 修改 x.data
error_caught = False
try:
    y.backward()
except RuntimeError as e:
    if "inplace" in str(e).lower() or "modified" in str(e).lower():
        error_caught = True
check("in-place modify detected", error_caught)

# Case 2: 正常使用（不修改）应正常工作
x2 = Variable(torch.FloatTensor([3.0]), requires_grad=True)
y2 = Square()(x2)
y2.backward()
check("normal backward still works", float_eq(x2.grad.data[0], 6.0))

# Case 3: mark_dirty — 声明 in-place 修改不报错
class InplaceScale(Function):
    """y = 2*x, done in-place"""
    def forward(self, x):
        self.save_for_backward(x)
        self.mark_dirty(x)
        for i in range(x.numel()):
            x._c.flat_set(i, x._c.flat_get(i) * 2.0)
        return x

    def backward(self, grad_output):
        x, = self.saved_tensors
        return grad_output * 2.0

x3 = Variable(torch.FloatTensor([3.0]), requires_grad=True)
y3 = InplaceScale()(x3)
check("mark_dirty: forward ok, y=6.0", float_eq(y3.data[0], 6.0))

# --- 16. Numerical gradient check ---
print("\n[Numerical gradient check: y = x^2]")

def numerical_grad(fn, x_val, eps=1e-3):
    x_plus = Variable(torch.FloatTensor([x_val + eps]))
    x_minus = Variable(torch.FloatTensor([x_val - eps]))
    return (fn(x_plus).data[0] - fn(x_minus).data[0]) / (2 * eps)

x_val = 3.0
x = Variable(torch.FloatTensor([x_val]), requires_grad=True)
y = Square()(x)
y.backward()
analytic = x.grad.data[0]
numeric = numerical_grad(lambda v: Square()(v), x_val)
check("analytic (%.4f) ~= numeric (%.4f)" % (to_scalar(analytic), to_scalar(numeric)),
      float_eq(analytic, numeric, eps=0.1))

# Check for chain: y = x^4
x_val = 2.0
x = Variable(torch.FloatTensor([x_val]), requires_grad=True)
h = Square()(x)
y = Square()(h)
y.backward()
analytic = x.grad.data[0]
numeric = numerical_grad(lambda v: Square()(Square()(v)), x_val)
check("chain x^4: analytic (%.4f) ~= numeric (%.4f)" % (to_scalar(analytic), to_scalar(numeric)),
      float_eq(analytic, numeric, eps=0.1))

# ============================================================
print("\n" + "=" * 60)
print("Results: %d passed, %d failed" % (tests_passed, tests_failed))
print("=" * 60)
exit(1 if tests_failed > 0 else 0)

