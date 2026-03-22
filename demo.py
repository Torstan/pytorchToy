"""
演示 TensorImpl → TensorBase → Tensor 三层结构

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

print("=" * 60)
print("1. 三层类结构")
print("=" * 60)
t = _C.Tensor([2, 3], 1.0)
print(f"type(t)           = {type(t)}")
print(f"isinstance Tensor = {isinstance(t, _C.Tensor)}")
print(f"isinstance Base   = {isinstance(t, _C.TensorBase)}")
print(f"Tensor 继承自 TensorBase: {issubclass(_C.Tensor, _C.TensorBase)}")

print(f"\n--- TensorBase 方法（元信息，不依赖算子定义）---")
print(f"t.dim()           = {t.dim()}")
print(f"t.numel()         = {t.numel()}")
print(f"t.sizes()         = {t.sizes()}")
print(f"t.strides()       = {t.strides()}")
print(f"t.defined()       = {t.defined()}")

print(f"\n--- Tensor 方法（算子，由代码生成器生成）---")
a = _C.Tensor([2, 3], 1.0)
b = _C.Tensor([2, 3], 2.0)
print(f"a.add(b)          = {a.add(b)}")
print(f"a.mul(b)          = {a.mul(b)}")
print(f"a + b             = {a + b}")

print("\n" + "=" * 60)
print("2. TensorImpl 引用计数共享")
print("=" * 60)
x = _C.Tensor([3], 5.0)
print(f"x = {x}")
print(f"x.use_count()     = {x.use_count()}")
print(f"x.data_ptr_id()   = {x.data_ptr_id():#x}")

# 把 Tensor 放入列表 → Python 侧多一个引用，引用计数 +1
holder = [x]
print(f"\nholder = [x] 后:")
print(f"x.use_count()     = {x.use_count()}")
print(f"同一个 TensorImpl? {x.data_ptr_id() == holder[0].data_ptr_id()}")

# 通过 holder[0] 修改数据会影响 x（因为共享 TensorImpl）
holder[0][0] = 99.0
print(f"\nholder[0][0] = 99.0 后:")
print(f"x[0] = {x[0]}  (x 也变了，因为共享 TensorImpl)")

del holder
print(f"\ndel holder 后:")
print(f"x.use_count()     = {x.use_count()}  (引用计数恢复)")

print("\n" + "=" * 60)
print("3. 为什么分 TensorBase 和 Tensor?")
print("=" * 60)
print("""
本项目通过 codegen.py 从 native_functions.yaml 自动生成算子代码，
模拟 PyTorch 的 torchgen 代码生成流程。

场景：在 native_functions.yaml 中新增/修改算子

只需重编译:
  - generated/*.h  (代码生成器重新生成)
  - bindings.o     (依赖 generated/*.h 和 ops.h)

不需要重编译:
  - tensor_base.o  (只依赖 TensorBase，不依赖算子定义)
  - tensor_impl.h  (底层数据结构)

验证方法: 运行 make demo_codegen，观察 tensor_base.o 不会被重编译。
""")

print("=" * 60)
print("4. 模拟前向传播")
print("=" * 60)
inp = _C.fill([1, 3], 1.0)
weight = _C.fill([3, 2], 0.5)
bias = _C.fill([1, 2], 0.1)

output = inp.matmul(weight)   # Tensor 方法（代码生成）
output = output.add(bias)     # Tensor 方法
output = output.relu()        # Tensor 方法
loss = output.sum()           # Tensor 方法

print(f"input   = {inp}")
print(f"weight  = {weight}")
print(f"output  = {output}")
print(f"loss    = {loss}")
