"""
演示 & 测试 TensorImpl → TensorBase → Tensor 三层结构

PyTorch 类层次:
    TensorImpl (c10/core/TensorImpl.h)
        └─ 持有真正的数据和元信息（sizes, strides, storage）
        └─ 引用计数对象，多个 Tensor 可共享同一个 TensorImpl

    TensorBase (aten/src/ATen/core/TensorBase.h)
        └─ 持有 intrusive_ptr<TensorImpl>（智能指针）
        └─ 只有元信息方法（dim, sizes, device 等）
        └─ 不依赖 native_functions.yaml → 修改算子不需要重编译

    Tensor (aten/src/ATen/templates/TensorBody.h)
        └─ 继承 TensorBase
        └─ 增加算子方法（add, mul, matmul 等）
        └─ 这些方法由 torchgen 从 native_functions.yaml 代码生成

构建流程:
    native_functions.yaml → codegen.py → generated/*.h → 编译
    修改 native_functions.yaml 后:
        tensor_base.o  不重新编译 (不依赖算子)
        bindings.o     重新编译   (依赖 generated/*.h)
"""
import _C


def approx(a, b, eps=1e-5):
    """浮点近似比较"""
    return abs(a - b) < eps


def tensor_data(t):
    """提取 tensor 所有逻辑元素为 list"""
    return [t[i].item() if t.dim() == 1 else t[i].item() for i in range(t.numel())]


def flat_values(t):
    """递归提取 tensor 所有元素为扁平 list"""
    if t.dim() == 0:
        return [t.item()]
    result = []
    for i in range(t.sizes()[0]):
        result.extend(flat_values(t[i]))
    return result


def assert_data(t, expected, msg=""):
    """断言 tensor 的逻辑数据与 expected 一致"""
    vals = flat_values(t)
    assert len(vals) == len(expected), \
        f"{msg}: length mismatch: got {len(vals)}, expected {len(expected)}"
    for i, (got, exp) in enumerate(zip(vals, expected)):
        assert approx(got, exp), \
            f"{msg}: index {i}: got {got}, expected {exp}"


# ============================================================
print("=" * 60)
print("0. 从嵌套 list 创建 1D / 2D / 3D tensor")
print("=" * 60)

# 1D tensor
t1 = _C.tensor([1.0, 2.0, 3.0])
assert t1.dim() == 1
assert t1.numel() == 3
assert t1.sizes() == [3]
assert t1.strides() == [1]
assert_data(t1, [1, 2, 3], "1D tensor")
print(f"  1D: {t1}  ✓")

# 2D tensor
t2 = _C.tensor([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])
assert t2.dim() == 2
assert t2.numel() == 6
assert t2.sizes() == [2, 3]
assert t2.strides() == [3, 1]
assert_data(t2, [1, 2, 3, 4, 5, 6], "2D tensor")
print(f"  2D: {t2}  ✓")

# 3D tensor
t3 = _C.tensor([[[1.0, 2.0],
                  [3.0, 4.0]],
                 [[5.0, 6.0],
                  [7.0, 8.0]]])
assert t3.dim() == 3
assert t3.numel() == 8
assert t3.sizes() == [2, 2, 2]
assert t3.strides() == [4, 2, 1]
print(f"  3D: {t3}  ✓")

# 多维索引
assert isinstance(t3[0, 0, 1], _C.Tensor), "完全索引应返回 0-dim Tensor"
assert t3[0, 0, 1].dim() == 0
assert approx(t3[0, 0, 1].item(), 2.0)

sub = t3[0, 1]  # 部分索引 → 1D view
assert isinstance(sub, _C.Tensor)
assert sub.dim() == 1
assert sub.sizes() == [2]
assert_data(sub, [3, 4], "t3[0,1]")

sub2 = t3[1]  # 单索引 → 2D view
assert sub2.dim() == 2
assert sub2.sizes() == [2, 2]
assert_data(sub2, [5, 6, 7, 8], "t3[1]")
print(f"  多维索引: t3[0,0,1]={t3[0,0,1]}, t3[0,1]={t3[0,1]}, t3[1]={t3[1]}  ✓")

# from_data
t3b = _C.Tensor.from_data([[[1, 2.0], [4, 5.0]]])
assert t3b.dim() == 3
assert t3b.sizes() == [1, 2, 2]
assert_data(t3b, [1, 2, 4, 5], "from_data")
print(f"  from_data: {t3b}  ✓")

# ============================================================
print("\n" + "=" * 60)
print("1. 三层类结构")
print("=" * 60)

t = _C.Tensor([2, 3], 1.0)
assert isinstance(t, _C.Tensor)
assert isinstance(t, _C.TensorBase)
assert issubclass(_C.Tensor, _C.TensorBase)
print(f"  Tensor 继承自 TensorBase  ✓")

# TensorBase 元信息方法
assert t.dim() == 2
assert t.numel() == 6
assert t.sizes() == [2, 3]
assert t.strides() == [3, 1]
assert t.defined() == True
print(f"  dim={t.dim()}, numel={t.numel()}, sizes={t.sizes()}, strides={t.strides()}  ✓")

# Tensor 算子方法
a = _C.Tensor([2, 3], 1.0)
b = _C.Tensor([2, 3], 2.0)
assert_data(a.add(b), [3]*6, "add")
assert_data(a.mul(b), [2]*6, "mul")
assert_data(a + b, [3]*6, "operator+")
assert_data(a * b, [2]*6, "operator*")
print(f"  add/mul/+/* 算子  ✓")

# ============================================================
print("\n" + "=" * 60)
print("2. TensorImpl 引用计数共享")
print("=" * 60)

x = _C.Tensor([3], 5.0)
assert x.use_count() == 1

# Python 中 x2 = x 只是别名（同一个 Python 对象），不会增加 C++ 引用计数
# 要创建一个新的 C++ Tensor 对象共享同一个 TensorImpl，需要 shallow_copy()
x2 = x.shallow_copy()
assert x.use_count() == 2, f"shallow_copy 后 use_count 应为 2，实际 {x.use_count()}"
assert x2.use_count() == 2
assert x.data_ptr_id() == x2.data_ptr_id(), "应共享同一个 TensorImpl"
print(f"  shallow_copy: use_count={x.use_count()}, 共享 TensorImpl  ✓")

# 也支持 copy.copy()
import copy
x3 = copy.copy(x)
assert x.use_count() == 3
print(f"  copy.copy: use_count={x.use_count()}  ✓")
del x3

# 通过 x2 修改数据会影响 x（共享 TensorImpl → 同一块 storage）
x2[0] = 99.0
assert approx(x[0].item(), 99.0), "共享 TensorImpl，修改应互相可见"
print(f"  数据互通: x2[0]=99 → x[0]={x[0].item()}  ✓")

del x2
assert x.use_count() == 1, "del x2 后 use_count 应恢复为 1"
print(f"  del x2: use_count={x.use_count()}  ✓")

# ============================================================
print("\n" + "=" * 60)
print("3. 模拟前向传播")
print("=" * 60)

inp = _C.fill([1, 3], 1.0)
weight = _C.fill([3, 2], 0.5)
bias = _C.fill([1, 2], 0.1)

output = inp.matmul(weight)   # [1,3] @ [3,2] = [1,2], 每个元素 = 1*0.5*3 = 1.5
assert output.sizes() == [1, 2]
assert_data(output, [1.5, 1.5], "matmul")

output = output.add(bias)     # 1.5 + 0.1 = 1.6
assert_data(output, [1.6, 1.6], "add bias")

output = output.relu()        # relu(1.6) = 1.6
assert_data(output, [1.6, 1.6], "relu")

loss = output.sum()            # 1.6 + 1.6 = 3.2
assert approx(loss, 3.2), f"sum: got {loss}, expected 3.2"
print(f"  matmul → add → relu → sum = {loss:.1f}  ✓")

# ============================================================
print("\n" + "=" * 60)
print("4. Storage / Stride / View 机制")
print("=" * 60)

# --- transpose ---
print("  --- transpose ---")
a = _C.Tensor([2, 3], 0.0)
for i in range(6):
    a[i] = float(i)
# a = [[0,1,2],[3,4,5]]

at = a.transpose(0, 1)
assert at.sizes() == [3, 2]
assert at.strides() == [1, 3]
assert at.is_contiguous() == False
assert a.storage_data_ptr() == at.storage_data_ptr(), "transpose 应共享 storage"
# transpose 后逻辑数据: [[0,3],[1,4],[2,5]]
assert_data(at, [0, 3, 1, 4, 2, 5], "transpose data")
print(f"  transpose: sizes={at.sizes()}, strides={at.strides()}, 共享storage  ✓")

# --- slice ---
print("  --- slice ---")
s = a.slice(1, 1, 3)  # 每行取第1、2列
assert s.sizes() == [2, 2]
assert s.strides() == [3, 1]
assert s.storage_offset() == 1
assert a.storage_data_ptr() == s.storage_data_ptr(), "slice 应共享 storage"
# slice 后逻辑数据: [[1,2],[4,5]]
assert_data(s, [1, 2, 4, 5], "slice data")
print(f"  slice: sizes={s.sizes()}, offset={s.storage_offset()}, 共享storage  ✓")

# --- reshape ---
print("  --- reshape ---")
r = a.reshape([3, 2])
assert r.sizes() == [3, 2]
assert r.strides() == [2, 1]
assert r.is_contiguous() == True
assert a.storage_data_ptr() == r.storage_data_ptr(), "reshape contiguous 应共享 storage"
assert_data(r, [0, 1, 2, 3, 4, 5], "reshape data")
print(f"  reshape: sizes={r.sizes()}, 零拷贝view  ✓")

# reshape with -1
r2 = a.reshape([-1])
assert r2.sizes() == [6]
assert_data(r2, [0, 1, 2, 3, 4, 5], "reshape -1")
print(f"  reshape([-1]): sizes={r2.sizes()}  ✓")

# --- expand ---
print("  --- expand ---")
v = _C.Tensor([1, 3], 0.0)
for i in range(3):
    v[i] = float(i + 1)
e = v.expand([4, 3])
assert e.sizes() == [4, 3]
assert e.strides() == [0, 1]
assert v.storage_data_ptr() == e.storage_data_ptr(), "expand 应共享 storage"
# expand 后每行都是 [1,2,3]
assert_data(e, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], "expand data")
print(f"  expand: sizes={e.sizes()}, strides={e.strides()}, 零拷贝广播  ✓")

# --- contiguous ---
print("  --- contiguous ---")
at_c = at.contiguous()
assert at_c.is_contiguous() == True
assert_data(at_c, [0, 3, 1, 4, 2, 5], "contiguous copy")
# 已经 contiguous 的 tensor 调用 contiguous 应共享 storage
a_c = a.contiguous()
assert a.storage_data_ptr() == a_c.storage_data_ptr(), "已 contiguous 不应拷贝"
print(f"  contiguous: non-contiguous 拷贝, contiguous 共享  ✓")

# --- 对 non-contiguous tensor 执行算子 ---
print("  --- 算子 on view tensors ---")
result = _C.add(at, at)
assert result.sizes() == [3, 2]
assert_data(result, [0, 6, 2, 8, 4, 10], "add(at, at)")

# matmul: a(2x3) @ a^T(3x2) = (2x2)
# [[0,1,2],[3,4,5]] @ [[0,3],[1,4],[2,5]] = [[5,14],[14,50]]
mt = _C.matmul(a, at)
assert mt.sizes() == [2, 2]
assert_data(mt, [5, 14, 14, 50], "matmul(a, a.T)")

# relu on non-contiguous
neg = _C.tensor([[-1.0, 2.0, -3.0],
                  [4.0, -5.0, 6.0]])
neg_t = neg.transpose(0, 1)  # [[-1,4],[2,-5],[-3,6]]
relu_result = neg_t.relu()
assert_data(relu_result, [0, 4, 2, 0, 0, 6], "relu on transposed")

# sum on non-contiguous
s_val = neg_t.sum()
assert approx(s_val, -1+4+2-5-3+6), f"sum on transposed: got {s_val}"
print(f"  add/matmul/relu/sum on view tensors  ✓")

# ============================================================
print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
