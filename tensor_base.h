#pragma once
#include "tensor_impl.h"
#include <sstream>

// ============================================================
// 模拟 c10::intrusive_ptr<TensorImpl>
// 对应 PyTorch: c10/util/intrusive_ptr.h
//
// 侵入式智能指针，引用计数存储在对象自身（TensorImpl::refcount_）中，
// 而非像 shared_ptr 那样额外分配控制块，减少一次内存分配。
// ============================================================

class IntrusivePtr {
    TensorImpl* ptr_ = nullptr;

public:
    IntrusivePtr() = default;
    explicit IntrusivePtr(TensorImpl* p) : ptr_(p) {}

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

    TensorImpl* get() const { return ptr_; }
    TensorImpl* operator->() const { return ptr_; }
    TensorImpl& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    int use_count() const { return ptr_ ? ptr_->refcount_.load() : 0; }
};

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
    IntrusivePtr impl_;

public:
    TensorBase() = default;
    explicit TensorBase(IntrusivePtr impl) : impl_(std::move(impl)) {}

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

    // 数据访问
    float* data_ptr() { return impl_->data_ptr(); }
    const float* data_ptr() const { return impl_->data_ptr(); }

    // 获取底层 TensorImpl
    TensorImpl* unsafeGetTensorImpl() const { return impl_.get(); }

    // 声明在此，实现在 tensor_base.cpp 中
    // 这样 TensorBase 拥有独立的编译单元，不依赖算子定义
    std::string repr() const;
};
