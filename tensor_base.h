#pragma once
#include "tensor_impl.h"
#include <sstream>

// ============================================================
// 模拟 c10::intrusive_ptr<T>
// 对应 PyTorch: c10/util/intrusive_ptr.h
//
// 通用侵入式智能指针模板。T 必须继承 IntrusivePtrTarget。
// 引用计数存储在对象自身中，避免像 shared_ptr 那样额外分配控制块。
// ============================================================

template<typename T>
class IntrusivePtr {
    T* ptr_ = nullptr;

public:
    IntrusivePtr() = default;
    explicit IntrusivePtr(T* p) : ptr_(p) {}

    IntrusivePtr(const IntrusivePtr& other) : ptr_(other.ptr_) {
        if (ptr_) ptr_->retain();
    }
    IntrusivePtr(IntrusivePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    IntrusivePtr& operator=(const IntrusivePtr& other) {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            if (ptr_) ptr_->retain();
        }
        return *this;
    }
    IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) ptr_->release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    ~IntrusivePtr() { if (ptr_) ptr_->release(); }

    T* get() const { return ptr_; }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    int use_count() const { return ptr_ ? ptr_->refcount_.load() : 0; }
};

// TensorImpl 专用的侵入式指针类型别名
using TensorImplPtr = IntrusivePtr<TensorImpl>;

// ============================================================
// TensorBase — 不依赖算子定义的基础句柄
// 对应 PyTorch: aten/src/ATen/core/TensorBase.h
//
// 只包含通用的元信息方法（dim, sizes, strides, device 等），
// 不包含任何算子方法（add, mul, matmul 等）。
// 这样修改 native_functions.yaml 时不需要重编译依赖 TensorBase 的代码。
// ============================================================

class TensorBase {
protected:
    TensorImplPtr impl_;

public:
    TensorBase() = default;
    explicit TensorBase(TensorImplPtr impl) : impl_(std::move(impl)) {}

    // 元信息访问 — 委托给 TensorImpl
    int dim() const { return impl_->dim(); }
    int numel() const { return impl_->numel(); }
    const std::vector<int>& sizes() const { return impl_->sizes_; }
    const std::vector<int>& strides() const { return impl_->strides_; }
    DeviceType device() const { return impl_->device_; }
    bool requires_grad() const { return impl_->requires_grad_; }
    void set_requires_grad(bool req) { impl_->requires_grad_ = req; }
    bool defined() const { return impl_ && impl_.get() != nullptr; }
    int use_count() const { return impl_.use_count(); }
    bool is_contiguous() const { return impl_->is_contiguous(); }
    int storage_offset() const { return impl_->storage_offset_; }
    const std::shared_ptr<Storage>& storage() const { return impl_->storage_; }

    // 数据访问
    float* data_ptr() { return impl_->data_ptr(); }
    const float* data_ptr() const { return impl_->data_ptr(); }

    // 获取底层 TensorImpl
    TensorImpl* unsafeGetTensorImpl() const { return impl_.get(); }

    // 声明在此，实现在 tensor_base.cpp 中
    // 这样 TensorBase 拥有独立的编译单元，不依赖算子定义
    std::string repr() const;
};
